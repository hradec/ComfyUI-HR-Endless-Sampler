"""CPU check: native LoRA parsing with independent H3 patch strengths."""

import importlib.util
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(ROOT)))
sys.argv.append("--cpu")
SPEC = importlib.util.spec_from_file_location("selective_lora", os.path.join(ROOT, "python", "selective_lora.py"))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class Model:
    """Record applied patches while retaining the native LoRA parser."""

    def __init__(self):
        """Start with an unchanged input model."""
        self.model = SimpleNamespace()
        self.applied = {}

    def clone(self):
        """Return an independent patch container."""
        return Model()

    def add_patches(self, values, strength):
        """Record exactly which model weights would be patched."""
        self.applied.update({key: strength for key in values})
        return list(values)


def main():
    """Check full, partial, bypass and invalid adapter handling."""
    targets = ["diffusion_model.blocks.0.attn.qkv_proj", "diffusion_model.blocks.0.mlp.fc1", "diffusion_model.token_refiner.blocks.0.attn.qkv_proj", "diffusion_model.token_refiner.blocks.0.mlp.fc1"]
    weights = {}
    for target in targets:
        weights[target + ".lora_A.weight"] = torch.ones(1, 2)
        weights[target + ".lora_B.weight"] = torch.ones(2, 1)
        weights[target + ".alpha"] = torch.tensor(1.0)
    keys = {target: target + ".weight" for target in targets}
    node = module.HREndlessSamplerSelectiveLora()
    model = Model()
    with patch.object(module.folder_paths, "get_full_path_or_raise", return_value="test.safetensors"), patch.object(module.comfy.utils, "load_torch_file", return_value=weights), patch.object(module.comfy.lora, "model_lora_keys_unet", return_value=keys):
        full = node.load_lora(model, "test", 1, 1, 1)[0]
        assert full.applied == {key: 1 for key in keys.values()}
        partial = node.load_lora(model, "test", 0.5, 0, -0.25)[0]
        assert partial.applied == {targets[0] + ".weight": 0.5, targets[2] + ".weight": -0.25, targets[3] + ".weight": -0.25}
        assert node.load_lora(model, "test", 0, 0, 0)[0] is model
        assert not model.applied
        with patch.object(module.comfy.lora, "load_lora", return_value={}):
            try:
                node.load_lora(model, "test", 1, 1, 1)
            except ValueError:
                pass
            else:
                raise AssertionError("An incompatible adapter must fail clearly")
    print("Selective LoRA checks passed (native parser, CPU).")


if __name__ == "__main__":
    main()
