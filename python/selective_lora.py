"""Selective H3 LoRA loading through ComfyUI's native patch machinery."""

import math
import re

import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths


class HREndlessSamplerSelectiveLora:
    """Apply independent strengths to the three H3 transformer weight groups."""

    @classmethod
    def INPUT_TYPES(cls):
        """Expose model, installed adapters and independent group strengths."""
        required = {"model": ("MODEL",), "lora_name": (folder_paths.get_filename_list("loras"),)}
        for name, description in (("attention", "Attention in the main transformer blocks."), ("feed_forward", "Feed-forward layers in the main transformer blocks."), ("token_refiner", "All attention and feed-forward layers in the token refiner.")):
            required[name] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": description + " 0 disables this portion; 1 applies full strength."})
        return {"required": required}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "model/loaders"
    DESCRIPTION = "Selectively load an H3 LoRA. Replace the normal loader for this adapter; do not load it twice. All strengths at 1 reproduce full loading. Partial loading can affect three-step generation quality."

    def load_lora(self, model, lora_name, attention, feed_forward, token_refiner):
        """Parse native LoRA patches once, then apply each group to a model clone."""
        strengths = {"attention": attention, "feed_forward": feed_forward, "token_refiner": token_refiner}
        if not all(math.isfinite(value) for value in strengths.values()):
            raise ValueError("LoRA strengths must be finite numbers.")
        if not any(strengths.values()):
            return (model,)

        # Use the installed ComfyUI converter and model key map, including alpha/rank.
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        weights = comfy.utils.load_torch_file(path, safe_load=True)
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        patches = comfy.lora.load_lora(comfy.lora_convert.convert_lora(weights), key_map)
        if not patches:
            raise ValueError("No compatible model LoRA weights found. Select an H3 model and H3 LoRA.")

        groups = {name: {} for name in strengths}
        for key, value in patches.items():
            # Classify resolved model keys, so supported input naming formats work alike.
            name = key[0] if isinstance(key, tuple) else key
            if re.search(r"(?:^|\.)token_refiner\.blocks\.\d+\.", name):
                group = "token_refiner"
            elif re.search(r"(?:^|\.)blocks\.\d+\.attn\.", name):
                group = "attention"
            elif re.search(r"(?:^|\.)blocks\.\d+\.mlp\.", name):
                group = "feed_forward"
            else:
                raise ValueError("Unsupported H3 LoRA target: " + name)
            groups[group][key] = value

        # Patch strengths scale updates without mutating the adapter or incoming model.
        result = model.clone()
        for group, selected in groups.items():
            if strengths[group] and selected:
                applied = result.add_patches(selected, strengths[group])
                if set(applied) != set(selected):
                    raise ValueError("Some selected H3 LoRA patches could not be applied: " + group)
        return (result,)
