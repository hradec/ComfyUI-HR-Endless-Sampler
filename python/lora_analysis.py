"""Read-only inspection of H3 LoRA adapters, and the endpoints that feed its UI.

Everything reported here is measured from the adapter file itself. Two tiers:

* the *header* tier reads only the safetensors header, so it costs about a
  millisecond and never touches tensor data. It answers provenance, which weight
  groups the adapter stores, how many parameters each group holds, and which
  transformer blocks are covered.
* the *applied* tier loads the weight tensors and measures how large each group's
  update actually is once ComfyUI's own ``strength * alpha / rank`` scaling is
  applied. That is a different question from parameter count, and for real
  adapters the two answers disagree sharply.

Neither tier says what a group *controls* during sampling. Attribution would
require a GPU ablation; the numbers here only describe where an adapter writes
and how hard.
"""

import json
import math
import os
import re
import struct
import threading
import time

import folder_paths

try:
    from aiohttp import web
except ImportError:  # aiohttp is present whenever the ComfyUI server is.
    web = None

try:
    from server import PromptServer
except ImportError:
    PromptServer = None


# The group table is the single source of truth. ``selective_lora`` imports it so
# that the loader and this report can never classify a weight differently.
# Order matters: the first matching pattern wins, exactly as the loader expects.
GROUP_PATTERNS = (
    ("token_refiner", re.compile(r"(?:^|\.)token_refiner\.blocks\.\d+\.")),
    ("attention", re.compile(r"(?:^|\.)blocks\.\d+\.attn\.")),
    ("feed_forward", re.compile(r"(?:^|\.)blocks\.\d+\.mlp\.")),
    ("modulation", re.compile(r"(?:^|\.)blocks\.\d+\.adaln_proj\.")),
)

GROUP_ORDER = ("attention", "feed_forward", "token_refiner", "modulation", "other")

# Artist-facing names and colors. Widgets are relabelled in the browser, so the
# stored widget names stay unchanged and saved workflows keep working.
GROUP_LABELS = {
    "attention": "Composition & Reference",
    "feed_forward": "Texture & Light",
    "token_refiner": "Prompt Reading",
    "modulation": "Conditioning Response",
    "other": "Everything Else",
}

GROUP_COLORS = {
    "attention": "#4aa3ff",
    "feed_forward": "#f0a03c",
    "token_refiner": "#b06cff",
    "modulation": "#3fbf8f",
    "other": "#8a8f98",
}

GROUP_SUMMARIES = {
    "attention": "how tokens read each other", 
    "feed_forward": "how each token is reshaped",
    "token_refiner": "how the prompt is read before the main stack",
    "modulation": "how strongly conditioning drives each block",
    "other": "targets outside the four groups above",
}

_BLOCK_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
_REFINER_RE = re.compile(r"(?:^|\.)token_refiner\.")

# Trailing markers that name the LoRA role rather than the module.
_ROLE_SUFFIXES = (
    ("alpha", (".alpha",)),
    ("A", (".lora_A.weight", ".lora_A", ".lora_down.weight", ".lora_down", ".lora.down.weight", ".lora.down")),
    ("B", (".lora_B.weight", ".lora_B", ".lora_up.weight", ".lora_up", ".lora.up.weight", ".lora.up")),
    ("mid", (".lora_mid.weight", ".lora_mid")),
)
_STRIP_SUFFIXES = tuple(sorted(
    {suffix for _, suffixes in _ROLE_SUFFIXES for suffix in suffixes} | {".weight", ".bias", ".default"},
    key=len, reverse=True))

# Bump when the applied-update math changes so stale cached results are dropped.
_DEEP_VERSION = 2
_MAX_HEADER_BYTES = 256 * 1024 * 1024

_ANALYSIS_LOCK = threading.Lock()
_HEADER_CACHE = {}
_DEEP_CACHE = {}
_DEEP_JOBS = {}


def classify_module(module_name):
    """Return the group a module name belongs to, defaulting to "other"."""
    for group, pattern in GROUP_PATTERNS:
        if pattern.search(module_name):
            return group
    return "other"


def _role(key):
    """Return the LoRA role a raw storage key carries, or None for other tensors."""
    for role, suffixes in _ROLE_SUFFIXES:
        if key.endswith(suffixes):
            return role
    return None


def _module_name(key):
    """Strip role and layer markers so A/B/alpha of one module share a name."""
    name = key
    for _ in range(4):
        trimmed = name
        for suffix in _STRIP_SUFFIXES:
            if trimmed.endswith(suffix):
                trimmed = trimmed[: -len(suffix)]
                break
        if trimmed == name:
            break
        name = trimmed
    return name


def _numel(shape):
    """Return the element count of a header shape, or 0 when it is unusable."""
    total = 1
    for extent in shape:
        total *= int(extent)
    return total


def _finite(value, cast=float, default=None):
    """Cast a value, returning the default when it is missing or not finite."""
    try:
        result = cast(value)
    except (TypeError, ValueError):
        return default
    if isinstance(result, float) and not math.isfinite(result):
        return default
    return result


def read_header(path):
    """Read only the safetensors header, without mapping any tensor data."""
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("Not a safetensors file: the length prefix is missing.")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= _MAX_HEADER_BYTES:
            raise ValueError("Not a safetensors file: the header length is implausible.")
        payload = handle.read(length)
        if len(payload) != length:
            raise ValueError("Not a safetensors file: the header is truncated.")
    header = json.loads(payload)
    metadata = header.pop("__metadata__", None) or {}
    return dict(metadata), header


def _collect_modules(header):
    """Group raw storage keys into modules with their A/B/alpha roles."""
    modules = {}
    for key in header:
        role = _role(key)
        if role is None:
            continue
        module = _module_name(key)
        if not module:
            continue
        modules.setdefault(module, {"keys": {}, "shapes": {}, "dtypes": {}})
        if role in modules[module]["keys"]:
            continue
        modules[module]["keys"][role] = key
        modules[module]["shapes"][role] = tuple(header[key].get("shape") or ())
        modules[module]["dtypes"][role] = header[key].get("dtype")
    return modules


def _module_params(module):
    """Count stored parameters for a module, excluding scalar alpha values."""
    total = 0
    for role in ("A", "B", "mid"):
        shape = module["shapes"].get(role)
        if shape:
            total += _numel(shape)
    return total


def _provenance(metadata):
    """Normalise the embedded training metadata into something readable."""
    software = metadata.get("software")
    trainer = metadata.get("trainer")
    if not trainer and isinstance(software, str):
        try:
            parsed = json.loads(software)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            version = parsed.get("version")
            trainer = parsed.get("name") or parsed.get("repo")
            if trainer and version:
                trainer = "{0} {1}".format(trainer, version)
    elif isinstance(software, str) and not trainer:
        trainer = software

    step = _finite(metadata.get("global_step"), int)
    epoch = None
    training_info = metadata.get("training_info")
    if isinstance(training_info, str):
        try:
            parsed_info = json.loads(training_info)
        except (TypeError, ValueError):
            parsed_info = None
        if isinstance(parsed_info, dict):
            epoch = _finite(parsed_info.get("epoch"), int)
            step = step if step is not None else _finite(parsed_info.get("step"), int)

    return {
        "trained_name": metadata.get("ss_output_name") or metadata.get("name"),
        "base_model": metadata.get("ss_base_model_version") or metadata.get("base_model"),
        "trainer": trainer,
        "training_mode": metadata.get("training_strategy"),
        "steps": step,
        "epoch": epoch,
        "rank": _finite(metadata.get("lora_rank"), int) or _finite(metadata.get("training_rank"), int),
        "first_frame_conditioning": _finite(metadata.get("first_frame_conditioning_p")),
        "source_format": metadata.get("source_format"),
        "task": metadata.get("task") or metadata.get("ss_task"),
        "raw": {key: value for key, value in metadata.items() if key != "training_info"},
    }


def _note_provenance(notes, provenance):
    """Add plain-language notes about what the embedded metadata does and does not say."""
    if not provenance["raw"]:
        notes.append("This adapter embeds no training metadata, so only structural facts are available. "
                     "The base model it was trained against is unknown.")
        return
    if provenance["training_mode"]:
        mode = str(provenance["training_mode"]).replace("_", " ")
        notes.append("Trained as {0}.".format(mode))
    conditioning = provenance["first_frame_conditioning"]
    if conditioning is not None and conditioning <= 0:
        notes.append("First-frame conditioning was disabled during training "
                     "(probability 0), so this adapter carries no training signal for how a "
                     "reference or first-frame image is read.")
    if provenance["rank"]:
        notes.append("Declared training rank {0}.".format(provenance["rank"]))
    if provenance["steps"]:
        step_text = "at step {0}".format(provenance["steps"])
        if provenance["epoch"]:
            step_text += " (epoch {0})".format(provenance["epoch"])
        notes.append("Training checkpoint {0}.".format(step_text))


def header_report(path, name=None):
    """Build the header-only report for one adapter file."""
    metadata, header = read_header(path)
    modules = _collect_modules(header)

    group_modules = {group: [] for group in GROUP_ORDER}
    groups = []
    block_params = {}
    ranks = set()
    dtypes = set()
    has_alpha = False
    unpaired = []
    total_params = 0

    for module, entry in modules.items():
        shapes = entry["shapes"]
        params = _module_params(entry)
        total_params += params
        for dtype in entry["dtypes"].values():
            if dtype:
                dtypes.add(dtype)
        if "alpha" in entry["keys"]:
            has_alpha = True
        if "A" in shapes and shapes["A"]:
            ranks.add(int(shapes["A"][0]))
            if "B" not in shapes:
                unpaired.append(module)
        elif "B" in shapes:
            unpaired.append(module)

        group = classify_module(module)
        group_modules[group].append((module, params))

        block_match = _BLOCK_RE.search(module)
        if block_match and not _REFINER_RE.search(module):
            index = int(block_match.group(1))
            block_params[index] = block_params.get(index, 0) + params

    for group in GROUP_ORDER:
        members = group_modules[group]
        params = sum(item[1] for item in members)
        groups.append({
            "id": group,
            "label": GROUP_LABELS[group],
            "color": GROUP_COLORS[group],
            "summary": GROUP_SUMMARIES[group],
            "modules": len(members),
            "params": params,
            "parameter_share": (params / total_params) if total_params else 0.0,
            "applied_share": None,
            "applied_norm": None,
            "scale": None,
            "dead": False,
        })

    ranked_blocks = sorted(block_params.items(), key=lambda item: -item[1])
    block_total = sum(block_params.values())
    blocks = {
        "count": len(block_params),
        "first": min(block_params) if block_params else None,
        "last": max(block_params) if block_params else None,
        "heaviest": [
            {"index": index, "share": (params / block_total) if block_total else 0.0}
            for index, params in ranked_blocks[:5]
        ],
        "lightest": [
            {"index": index, "share": (params / block_total) if block_total else 0.0}
            for index, params in ranked_blocks[-5:]
        ],
    }

    notes = []
    _note_provenance(notes, _provenance(metadata))
    notes.append(
        "ComfyUI applies each update at full scale (x1.0) because no alpha values are stored."
        if not has_alpha else
        "ComfyUI scales each update by its stored alpha divided by its rank."
    )
    empty = [GROUP_LABELS[group] for group in GROUP_ORDER
             if group != "other" and not group_modules[group]]
    if empty:
        notes.append("No {0} weights are stored, so those sliders have nothing to act on.".format(
            ", ".join(empty)))
    if unpaired:
        notes.append("{0} module(s) store only one half of a LoRA pair and cannot be applied.".format(
            len(unpaired)))

    return {
        "ok": True,
        "name": name or os.path.basename(path),
        "file": {
            "path": path,
            "bytes": os.path.getsize(path),
            "modified_ms": int(os.stat(path).st_mtime * 1000),
        },
        "metric": "parameters",
        "groups": groups,
        "blocks": blocks,
        "provenance": _provenance(metadata),
        "structure": {
            "tensors": len(header),
            "modules": len(modules),
            "params": total_params,
            "ranks": sorted(ranks),
            "dtypes": sorted(dtypes),
            "has_alpha": has_alpha,
        },
        "notes": notes,
    }


def deep_report(path, name=None):
    """Measure the applied update of every group, using ComfyUI's own scaling."""
    import torch
    from safetensors import safe_open

    torch.set_grad_enabled(False)
    started = time.perf_counter()
    metadata, header = read_header(path)
    modules = _collect_modules(header)

    energies = {}
    params = {}
    block_energy = {}
    scales = {}
    dead = {}
    effective_rank = {}
    worst_top1 = None
    skipped = 0

    with safe_open(path, framework="pt") as handle:
        available = set(handle.keys())
        for module, entry in modules.items():
            keys = entry["keys"]
            shapes = entry["shapes"]
            if "A" not in keys or "B" not in keys:
                skipped += 1
                continue
            if "A" not in shapes or "B" not in shapes or not shapes["A"] or not shapes["B"]:
                skipped += 1
                continue
            if keys["A"] not in available or keys["B"] not in available:
                skipped += 1
                continue

            group = classify_module(module)
            params[group] = params.get(group, 0) + _module_params(entry)

            a = handle.get_tensor(keys["A"]).to(torch.float32)
            b = handle.get_tensor(keys["B"]).to(torch.float32)
            if a.shape[0] != b.shape[1]:
                # Only the [out, rank] x [rank, in] layout is supported; anything
                # else is counted structurally but left out of the update metric.
                skipped += 1
                continue

            # ||BA||_F^2 = trace((B^T B)(A A^T)). Both Grams are only rank-sized, so
            # the [out, in] update is never materialised. The product of two Gram
            # matrices is not symmetric, so one Gram is whitened by its square root:
            # that is a similarity transform, leaving both the trace and every
            # eigenvalue unchanged while allowing a symmetric solver.
            gram_b = b.t() @ b
            gram_a = a @ a.t()
            weights_a, vectors_a = torch.linalg.eigh(gram_a)
            root_a = (vectors_a * weights_a.clamp(min=0).sqrt()) @ vectors_a.t()
            symmetrized = root_a @ gram_b @ root_a
            eigenvalues = torch.linalg.eigvalsh((symmetrized + symmetrized.t()) * 0.5).clamp(min=0)
            energy = float(eigenvalues.sum())

            rank = a.shape[0]
            scale = 1.0
            alpha_key = keys.get("alpha")
            if alpha_key and alpha_key in available:
                alpha_value = _finite(handle.get_tensor(alpha_key).reshape(-1)[0].item())
                if alpha_value is not None:
                    scale = alpha_value / rank if rank else 1.0
            scales.setdefault(group, set()).add(round(scale, 6))

            scaled_energy = energy * scale * scale
            energies[group] = energies.get(group, 0.0) + scaled_energy
            if scaled_energy <= 0.0:
                dead[group] = dead.get(group, 0) + 1

            block_match = _BLOCK_RE.search(module)
            if block_match and not _REFINER_RE.search(module):
                index = int(block_match.group(1))
                block_energy[index] = block_energy.get(index, 0.0) + scaled_energy

            if energy > 0:
                total = float(eigenvalues.sum())
                top1 = float(eigenvalues[-1]) / total if total else 0.0
                concentration = (total * total) / float((eigenvalues ** 2).sum()) if total else 0.0
                current = effective_rank.get(group)
                if current is None or concentration < current["min"]:
                    effective_rank[group] = {
                        "min": round(concentration, 2),
                        "min_module": module.split(".")[-1],
                        "block": int(block_match.group(1)) if block_match and not _REFINER_RE.search(module) else None,
                    }
                if worst_top1 is None or top1 > worst_top1["top1"]:
                    worst_top1 = {"top1": round(top1, 4), "block": int(block_match.group(1)) if block_match and not _REFINER_RE.search(module) else None}

    total_energy = sum(energies.values())
    groups = []
    for group in GROUP_ORDER:
        energy = energies.get(group, 0.0)
        groups.append({
            "id": group,
            "label": GROUP_LABELS[group],
            "color": GROUP_COLORS[group],
            "modules": len([m for m in modules if classify_module(m) == group]),
            "params": params.get(group, 0),
            "applied_norm": energy ** 0.5 if energy > 0 else 0.0,
            "applied_share": (energy / total_energy) if total_energy else 0.0,
            "scale": sorted(scales[group])[0] if len(scales.get(group, ())) == 1 else None,
            "dead_modules": dead.get(group, 0),
            "dead": bool(group in params and energies.get(group, 0.0) <= 0.0),
            "effective_rank": effective_rank.get(group),
        })

    ranked_blocks = sorted(block_energy.items(), key=lambda item: -item[1])
    block_total = sum(block_energy.values())
    blocks = {
        "count": len(block_energy),
        "heaviest": [
            {"index": index, "share": (energy / block_total) if block_total else 0.0}
            for index, energy in ranked_blocks[:5]
        ],
        "lightest": [
            {"index": index, "share": (energy / block_total) if block_total else 0.0}
            for index, energy in ranked_blocks[-5:]
        ],
        "first_half_share": (
            sum(energy for index, energy in block_energy.items() if index < 25) / block_total
            if block_total else 0.0
        ),
    }

    notes = []
    for group in groups:
        if group["dead"] and group["modules"]:
            notes.append(
                "{0}: {1} module(s) store weights, but the update they apply is exactly zero, "
                "so this slider does nothing on this adapter.".format(group["label"], group["modules"]))
    if total_energy > 0:
        heaviest = max(groups, key=lambda item: item["applied_share"])
        notes.append("Compared against the adapter's own total update, {0} carries the most "
                     "influence ({1:.0f}%).".format(heaviest["label"], 100 * heaviest["applied_share"]))
    if worst_top1 and worst_top1["top1"] >= 0.9:
        notes.append("One block's update is almost a single direction (top component "
                     "{0:.0f}%{1}), which can behave like a blunt global edit.".format(
                         100 * worst_top1["top1"],
                         " on block {0}".format(worst_top1["block"]) if worst_top1["block"] is not None else ""))
    if skipped:
        notes.append("{0} module(s) were left out of the update measurement.".format(skipped))

    return {
        "ok": True,
        "metric": "applied_update",
        "groups": groups,
        "blocks": blocks,
        "notes": notes,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


def _resolve(name):
    """Map a LoRA list entry to a real path, refusing anything outside the folders."""
    if not isinstance(name, str) or not name:
        raise ValueError("A LoRA name is required.")
    if name not in folder_paths.get_filename_list("loras"):
        raise ValueError("Unknown LoRA: {0}".format(name))
    path = folder_paths.get_full_path_or_raise("loras", name)
    if not os.path.isfile(path):
        raise ValueError("The LoRA file could not be read: {0}".format(name))
    return path


def _stat_key(path):
    """Identify a file version, so an edited adapter is never reported from cache."""
    stat = os.stat(path)
    return (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)


def analyze(name, deep=False):
    """Return the current report for an adapter, starting a background job if needed."""
    path = _resolve(name)
    key = _stat_key(path)
    with _ANALYSIS_LOCK:
        cached_header = _HEADER_CACHE.get(key)
    if cached_header is None:
        cached_header = header_report(path, name)
        with _ANALYSIS_LOCK:
            _HEADER_CACHE.clear()
            _HEADER_CACHE[key] = cached_header

    report = dict(cached_header)
    report["deep"] = {"state": "skipped", "error": None}

    if not deep:
        return report

    deep_key = (key, _DEEP_VERSION)
    with _ANALYSIS_LOCK:
        cached_deep = _DEEP_CACHE.get(deep_key)
        job = _DEEP_JOBS.get(deep_key)
        if cached_deep is None and job is None:
            _DEEP_JOBS[deep_key] = {"state": "computing", "error": None}
            job = _DEEP_JOBS[deep_key]
            job["thread"] = threading.Thread(
                target=_run_deep, args=(deep_key, path, name), name="hr_endless_lora_analysis", daemon=True)
            start = job["thread"]
        else:
            start = None

    if cached_deep is not None:
        report.update({
            "metric": "applied_update",
            "groups": _merge_groups(cached_header["groups"], cached_deep["groups"]),
            "blocks": cached_deep["blocks"],
            "notes": cached_header["notes"] + cached_deep["notes"],
            "deep": {"state": "ready", "error": None, "elapsed_ms": cached_deep["elapsed_ms"]},
        })
    else:
        report["deep"] = {"state": job["state"], "error": job["error"]}

    if start is not None:
        start.start()
    return report


def _merge_groups(header_groups, deep_groups):
    """Combine parameter counts with measured update sizes for the UI."""
    deep_by_id = {group["id"]: group for group in deep_groups}
    merged = []
    for group in header_groups:
        entry = dict(group)
        measured = deep_by_id.get(group["id"])
        if measured:
            for field in ("applied_share", "applied_norm", "scale", "dead", "dead_modules", "effective_rank"):
                entry[field] = measured.get(field)
        merged.append(entry)
    return merged


def _run_deep(deep_key, path, name):
    """Compute the applied-update tier on a worker thread and cache the result."""
    try:
        result = deep_report(path, name)
        state = {"state": "ready", "error": None}
    except Exception as error:  # A broken adapter must not wedge the endpoint.
        result = None
        state = {"state": "failed", "error": "{0}: {1}".format(type(error).__name__, error)}
    with _ANALYSIS_LOCK:
        if result is not None:
            _DEEP_CACHE[deep_key] = result
        _DEEP_JOBS[deep_key] = state


def reset_cache():
    """Forget every cached report. Used by tests and diagnostics."""
    with _ANALYSIS_LOCK:
        _HEADER_CACHE.clear()
        _DEEP_CACHE.clear()
        _DEEP_JOBS.clear()


_SERVER = None if PromptServer is None else getattr(PromptServer, "instance", None)

if _SERVER is not None and web is not None:
    @_SERVER.routes.get("/hr_endless_sampler_lora/analysis")
    async def hr_endless_sampler_lora_analysis(request):
        """Report what a LoRA contains, without loading it onto a model."""
        name = request.rel_url.query.get("name", "")
        deep = request.rel_url.query.get("deep", "0") not in ("0", "false", "")
        try:
            # Header parsing and the deep job scheduling are both cheap; the
            # tensor work runs on a worker thread so the server never blocks.
            report = analyze(name, deep=deep)
        except (OSError, ValueError, ImportError, KeyError) as error:
            return web.json_response({"ok": False, "error": str(error)}, status=400)
        return web.json_response(report, headers={"Cache-Control": "no-store"})

    @_SERVER.routes.get("/hr_endless_sampler_lora/groups")
    async def hr_endless_sampler_lora_groups(request):
        """Describe the group scheme so the browser never hard-codes it."""
        return web.json_response({
            "order": list(GROUP_ORDER),
            "labels": GROUP_LABELS,
            "colors": GROUP_COLORS,
            "summaries": GROUP_SUMMARIES,
        }, headers={"Cache-Control": "no-store"})
