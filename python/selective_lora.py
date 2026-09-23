"""Selective H3 LoRA loading through ComfyUI's native patch machinery."""

import math

import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths

try:
    from .lora_analysis import classify_module
except ImportError:  # Loaded straight from its file path, as the tests do.
    from lora_analysis import classify_module


class HREndlessSamplerSelectiveLora:
    """Apply independent strengths to H3 LoRA weight groups."""

    @classmethod
    def INPUT_TYPES(cls):
        """Expose model, installed adapters and independent group strengths."""
        required = {"model": ("MODEL",), "lora_name": (folder_paths.get_filename_list("loras"),)}
        for name, description in (
            ("attention", "Composition & Reference. How tokens read each other: framing, subject placement and how closely the result follows reference images. Lowering it is a fair camera-drift experiment, but it can also weaken identity and continuity, and it is not a dedicated camera control."),
            ("feed_forward", "Texture & Light. How each token is reshaped on its own: texture, detail, lighting and finish. It carries composition too, so this does not cleanly separate look from camera."),
            ("token_refiner", "Prompt Reading. How the text prompt is read before the main transformer. Changes can affect how strongly the prompt is understood; this is not a style or camera control."),
        ):
            required[name] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": description + " 0 disables this portion; 1 applies full strength."})
        # Optional inputs preserve existing workflows; native-mapped extra targets stay controllable.
        optional = {}
        for name, description in (
            ("modulation", "Conditioning Response. How strongly conditioning scales, shifts and gates each block as it denoises. The effect depends on how the adapter was trained, and some adapters store these weights but apply nothing at all; the group bars show which. Setting 0 skips these patches, including adapters with incompatible shapes; a nonzero strength cannot repair incompatible shapes."),
            ("other", "Everything Else. Targets outside the four groups above, such as final output modulation. Usually empty for H3 adapters. When present, its effect depends entirely on what the adapter actually stored."),
        ):
            optional[name] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": description + " 0 disables this portion; 1 applies full strength."})
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "model/loaders"
    DESCRIPTION = "Selectively load an H3 LoRA, with a live chart of what the adapter actually contains. Replace the normal loader for this adapter; do not load it twice. All strengths at 1 pass all native-mapped patches through at full strength. Groups overlap in effect according to training, so they are not separate image versus sampling-step roles, and only targets present in the adapter are affected. Partial loading of an acceleration LoRA can impair low-step generation. Compare strengths with the same seed, prompt, sampler and steps."

    def load_lora(self, model, lora_name, attention, feed_forward, token_refiner, modulation=1.0, other=1.0):
        """Parse native LoRA patches once, then apply each group to a model clone."""
        strengths = {"attention": attention, "feed_forward": feed_forward, "token_refiner": token_refiner, "modulation": modulation, "other": other}
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
            # Classify resolved model keys through the shared group table, so the
            # report shown in the browser and this loader can never disagree.
            resolved = key[0] if isinstance(key, tuple) else key
            groups[classify_module(resolved)][key] = value

        # Patch strengths scale updates without mutating the adapter or incoming model.
        result = model.clone()
        for group, selected in groups.items():
            if strengths[group] and selected:
                applied = result.add_patches(selected, strengths[group])
                if set(applied) != set(selected):
                    raise ValueError("Some selected H3 LoRA patches could not be applied: " + group)
        return (result,)
