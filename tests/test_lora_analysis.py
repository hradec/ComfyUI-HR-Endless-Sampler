"""CPU check: LoRA inspection metrics, group classification and provenance.

The synthetic adapter carries hand-computed expected values, so the applied-update
metric is checked against arithmetic rather than against its own implementation.
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import types

import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(ROOT)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.argv.append("--cpu")
SPEC = importlib.util.spec_from_file_location("lora_analysis", os.path.join(ROOT, "python", "lora_analysis.py"))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

LORA_DIR = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "models", "loras")
H3_ADAPTERS = (
    "minimax_h3_lms_v1.0_r64.safetensors",
    "h3-realism-people-t2v-i2v-r2v.safetensors",
    "TaoMate-H3-3step-ComfyUI.safetensors",
)


def close(left, right, tolerance=1e-6):
    """Compare two floats without depending on exact binary representation."""
    assert abs(left - right) <= tolerance, "{0} != {1}".format(left, right)


def build_synthetic(directory):
    """Write an adapter whose per-group update sizes are known by hand."""
    tensors = {}
    # rank 4 throughout; module updates are chosen so ||B A||_F^2 is exact.
    tensors["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"] = torch.ones(4, 6)
    tensors["diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"] = torch.ones(5, 4)
    tensors["diffusion_model.blocks.0.attn.qkv_proj.alpha"] = torch.tensor(4.0)
    # alpha 4 / rank 4 => scale 1.0 ; B A = 4 * ones(5, 6) => 30 * 16 = 480

    tensors["diffusion_model.blocks.1.mlp.fc1.lora_A.weight"] = torch.ones(4, 6)
    tensors["diffusion_model.blocks.1.mlp.fc1.lora_B.weight"] = torch.full((5, 4), 2.0)
    # no alpha => scale 1.0 ; B A = 8 * ones(5, 6) => 30 * 64 = 1920

    tensors["diffusion_model.token_refiner.blocks.0.attn.out_proj.lora_A.weight"] = torch.ones(4, 2)
    tensors["diffusion_model.token_refiner.blocks.0.attn.out_proj.lora_B.weight"] = torch.ones(3, 4)
    tensors["diffusion_model.token_refiner.blocks.0.attn.out_proj.alpha"] = torch.tensor(8.0)
    # alpha 8 / rank 4 => scale 2.0 ; B A = 4 * ones(3, 2) => 6 * 16 = 96, scaled => 384

    tensors["diffusion_model.blocks.2.adaln_proj.linear.lora_A.weight"] = torch.ones(4, 2)
    tensors["diffusion_model.blocks.2.adaln_proj.linear.lora_B.weight"] = torch.zeros(5, 4)
    # an entirely zero update: stored, but applies nothing

    tensors["diffusion_model.final_layer.adaln_proj.linear.lora_A.weight"] = torch.ones(4, 2)
    tensors["diffusion_model.final_layer.adaln_proj.linear.lora_B.weight"] = torch.ones(2, 4)
    # outside every named group => "other" ; B A = 4 * ones(2, 2) => 4 * 16 = 64

    metadata = {
        "name": "synthetic_sharpness",
        "ss_base_model_version": "minimax_h3_ref2va",
        "software": json.dumps({"name": "ai-toolkit", "version": "0.13.4"}),
        "training_strategy": "text_to_video",
        "first_frame_conditioning_p": "0.0",
        "global_step": "1500",
        "training_info": json.dumps({"epoch": 3, "step": 1500}),
    }
    path = os.path.join(directory, "synthetic_rank4.safetensors")
    save_file(tensors, path, metadata=metadata)
    return path


def check_classification():
    """Pin the group table against all three on-disk H3 key conventions."""
    cases = {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": "attention",
        "blocks.0.attn.qkv_proj.lora_A": "attention",
        "diffusion_model.blocks.7.attn.out_proj": "attention",
        "diffusion_model.blocks.0.mlp.fc2.lora_B.weight": "feed_forward",
        "diffusion_model.blocks.0.adaln_proj.linear.lora_B.weight": "modulation",
        # The refiner nests attn and mlp modules; it must win over both.
        "diffusion_model.token_refiner.blocks.0.attn.qkv_proj.lora_A.weight": "token_refiner",
        "diffusion_model.token_refiner.blocks.1.mlp.fc1.lora_B.weight": "token_refiner",
        # Output-side modulation is not a per-block target, so it stays in "other".
        "diffusion_model.final_layer.adaln_proj.linear.lora_A.weight": "other",
        "diffusion_model.video_out.weight": "other",
    }
    for key, expected in cases.items():
        actual = module.classify_module(key)
        assert actual == expected, "{0} classified as {1}, expected {2}".format(key, actual, expected)
    # Non-H3 key shapes must not be mistaken for H3 groups.
    assert module.classify_module("lora_unet_single_blocks_13_linear1.lora_down") == "other"


def check_synthetic(path):
    """Verify parameter shares, applied shares, alpha scaling and dead detection."""
    header = module.header_report(path, "synthetic_rank4.safetensors")
    groups = {group["id"]: group for group in header["groups"]}

    assert header["structure"]["modules"] == 5, header["structure"]
    assert header["structure"]["ranks"] == [4]
    assert header["structure"]["has_alpha"] is True
    assert header["blocks"]["count"] == 3, header["blocks"]["count"]

    expected_params = {"attention": 44, "feed_forward": 44, "token_refiner": 20, "modulation": 28, "other": 16}
    total_params = 152
    for group, params in expected_params.items():
        assert groups[group]["params"] == params, (group, groups[group]["params"], params)
        close(groups[group]["parameter_share"], params / total_params)

    # The header tier must not claim an influence metric it has not measured.
    assert header["metric"] == "parameters"
    assert all(group["applied_share"] is None for group in header["groups"])

    provenance = header["provenance"]
    assert provenance["trained_name"] == "synthetic_sharpness"
    assert provenance["base_model"] == "minimax_h3_ref2va"
    assert provenance["trainer"] == "ai-toolkit 0.13.4"
    assert provenance["steps"] == 1500 and provenance["epoch"] == 3
    assert any("First-frame conditioning was disabled" in note for note in header["notes"])

    deep = module.deep_report(path, "synthetic_rank4.safetensors")
    assert deep["metric"] == "applied_update"
    measured = {group["id"]: group for group in deep["groups"]}

    # scale = stored alpha / rank when alpha exists, otherwise 1.0.
    close(measured["attention"]["scale"], 1.0)
    close(measured["feed_forward"]["scale"], 1.0)
    close(measured["token_refiner"]["scale"], 2.0)

    expected_energy = {"attention": 480.0, "feed_forward": 1920.0, "token_refiner": 384.0, "modulation": 0.0, "other": 64.0}
    total_energy = sum(expected_energy.values())
    for group, energy in expected_energy.items():
        close(measured[group]["applied_norm"] ** 2, energy, tolerance=1e-3)
        close(measured[group]["applied_share"], energy / total_energy, tolerance=1e-6)

    # A stored-but-inert group is the whole point of reporting both metrics.
    assert measured["modulation"]["dead"] is True
    assert measured["modulation"]["dead_modules"] == 1
    assert measured["modulation"]["applied_share"] == 0.0
    # Its parameter footprint stays large even though its influence is nil.
    assert groups["modulation"]["parameter_share"] > 0.18
    assert any("does nothing" in note for note in deep["notes"])

    # Cross-check the Gram identity against the naive product of the stored tensors.
    pairs = (
        ("attention", "diffusion_model.blocks.0.attn.qkv_proj", 1.0),
        ("feed_forward", "diffusion_model.blocks.1.mlp.fc1", 1.0),
        ("token_refiner", "diffusion_model.token_refiner.blocks.0.attn.out_proj", 2.0),
        ("other", "diffusion_model.final_layer.adaln_proj.linear", 1.0),
    )
    from safetensors import safe_open
    naive = {group: 0.0 for group in expected_energy}
    with safe_open(path, framework="pt") as handle:
        for group, base, scale in pairs:
            a = handle.get_tensor(base + ".lora_A.weight").to(torch.float32)
            b = handle.get_tensor(base + ".lora_B.weight").to(torch.float32)
            naive[group] = float(torch.linalg.matrix_norm(b @ a) ** 2) * scale * scale
    # The naive product must agree with the Gram-based metric for every group.
    for group, energy in naive.items():
        close(measured[group]["applied_norm"] ** 2, energy, tolerance=1e-3)
    close(naive["attention"], 480.0, tolerance=1e-3)
    close(naive["feed_forward"], 1920.0, tolerance=1e-3)
    return header, deep


def check_real_adapters():
    """Run the analyzer against the real H3 adapters when they are installed."""
    if not os.path.isdir(LORA_DIR):
        print("  (skipping real adapters: {0} is absent)".format(LORA_DIR))
        return
    checked = 0
    for name in H3_ADAPTERS:
        path = os.path.join(LORA_DIR, name)
        if not os.path.isfile(path):
            continue
        header = module.header_report(path, name)
        ids = [group["id"] for group in header["groups"]]
        assert ids == list(module.GROUP_ORDER), ids
        assert header["structure"]["modules"] > 0
        assert header["structure"]["tensors"] > 0
        # Every group must account for its share of the file exactly once.
        close(sum(group["parameter_share"] for group in header["groups"]), 1.0, tolerance=1e-9)
        checked += 1

    if checked:
        # The library's known dead-AdaLN adapter must be reported as inert.
        lms = os.path.join(LORA_DIR, "minimax_h3_lms_v1.0_r64.safetensors")
        if os.path.isfile(lms):
            deep = module.deep_report(lms, "minimax_h3_lms_v1.0_r64.safetensors")
            groups = {group["id"]: group for group in deep["groups"]}
            assert groups["modulation"]["dead"] is True, groups["modulation"]
            assert groups["modulation"]["applied_share"] == 0.0
            # Parameter footprint and applied footprint must disagree sharply here.
            assert groups["modulation"]["params"] > groups["attention"]["params"]
            assert groups["attention"]["applied_share"] > groups["modulation"]["applied_share"]
            assert any("does nothing" in note for note in deep["notes"])
    print("  real adapters checked: {0}".format(checked))


def check_routes():
    """Verify the HTTP surface registers and answers correctly without a server."""
    import asyncio
    from types import SimpleNamespace

    registered = {}

    class Routes:
        """Record route handlers instead of attaching them to an aiohttp app."""

        def get(self, path):
            """Register a GET handler under its path."""
            def decorate(handler):
                registered[path] = handler
                return handler
            return decorate

        post = get

    class StubPromptServer:
        """Stand in for the ComfyUI server module during a fresh import."""
        instance = SimpleNamespace(routes=Routes())

    stub = types.ModuleType("server")
    stub.PromptServer = StubPromptServer
    saved = sys.modules.get("server")
    sys.modules["server"] = stub
    try:
        spec = importlib.util.spec_from_file_location(
            "lora_analysis_routes", os.path.join(ROOT, "python", "lora_analysis.py"))
        routes_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(routes_module)
    finally:
        if saved is not None:
            sys.modules["server"] = saved
        else:
            sys.modules.pop("server", None)

    assert "/hr_endless_sampler_lora/analysis" in registered, sorted(registered)
    assert "/hr_endless_sampler_lora/groups" in registered, sorted(registered)

    directory = tempfile.mkdtemp(prefix="hr_endless_routes_")
    original_list = routes_module.folder_paths.get_filename_list
    original_full = routes_module.folder_paths.get_full_path_or_raise
    try:
        path = build_synthetic(directory)
        routes_module.folder_paths.get_filename_list = lambda _kind: ["synthetic_rank4.safetensors"]
        routes_module.folder_paths.get_full_path_or_raise = lambda _kind, _name: path
        handler = registered["/hr_endless_sampler_lora/analysis"]

        def request(**query):
            """Build the only part of an aiohttp request these handlers read."""
            return SimpleNamespace(rel_url=SimpleNamespace(query=query))

        response = asyncio.run(handler(request(name="synthetic_rank4.safetensors", deep="0")))
        assert response.status == 200, response.status
        body = json.loads(response.text)
        assert body["ok"] is True and len(body["groups"]) == 5, body.keys()
        assert body["deep"]["state"] == "skipped"

        # An unknown or traversing name must be refused, never resolved.
        for bad_name in ("../../etc/passwd", "not_a_real_lora.safetensors", ""):
            refused = asyncio.run(handler(request(name=bad_name)))
            assert refused.status == 400, (bad_name, refused.status)
            assert json.loads(refused.text)["ok"] is False

        catalog = asyncio.run(registered["/hr_endless_sampler_lora/groups"](request()))
        described = json.loads(catalog.text)
        assert described["order"][0] == "attention"
        assert described["labels"]["attention"] == "Composition & Reference"
    finally:
        routes_module.folder_paths.get_filename_list = original_list
        routes_module.folder_paths.get_full_path_or_raise = original_full
        shutil.rmtree(directory, ignore_errors=True)


def main():
    """Check classification, the synthetic metric arithmetic, routes and real adapters."""
    check_classification()
    directory = tempfile.mkdtemp(prefix="hr_endless_lora_")
    try:
        path = build_synthetic(directory)
        check_synthetic(path)
        # The route resolves names through folder_paths; drive it through that path.
        original_list = module.folder_paths.get_filename_list
        original_full = module.folder_paths.get_full_path_or_raise
        module.folder_paths.get_filename_list = lambda _kind: ["synthetic_rank4.safetensors"]
        module.folder_paths.get_full_path_or_raise = lambda _kind, _name: path
        try:
            module.reset_cache()
            report = module.analyze("synthetic_rank4.safetensors", deep=False)
            assert report["ok"] is True and report["deep"]["state"] == "skipped"
            running = module.analyze("synthetic_rank4.safetensors", deep=True)
            assert running["deep"]["state"] in ("computing", "ready"), running["deep"]
            assert module.analyze("synthetic_rank4.safetensors", deep=True)["ok"] is True
            for _ in range(600):
                done = module.analyze("synthetic_rank4.safetensors", deep=True)
                if done["deep"]["state"] != "computing":
                    break
                time.sleep(0.05)
            assert done["deep"]["state"] == "ready", done["deep"]
            assert done["metric"] == "applied_update"
            close(sum(group["applied_share"] for group in done["groups"]), 1.0, tolerance=1e-6)
            assert done["groups"][3]["dead"] is True
            try:
                module.analyze("not_a_real_lora.safetensors", deep=False)
            except ValueError:
                pass
            else:
                raise AssertionError("An unknown adapter name must be rejected")
        finally:
            module.folder_paths.get_filename_list = original_list
            module.folder_paths.get_full_path_or_raise = original_full
            module.reset_cache()
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    check_real_adapters()
    check_routes()
    print("LoRA analysis checks passed (CPU).")


if __name__ == "__main__":
    main()
