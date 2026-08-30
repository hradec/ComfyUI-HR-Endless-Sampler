"""Shared local-model selection for HR Endless Sampler directors."""

from dataclasses import dataclass
from pathlib import Path

import folder_paths


DIRECTOR_BACKENDS = ("gemma4", "qwen3.5")
_MODEL_ROOTS = ("llama_cpp", "LLM/GGUF")


class DirectorConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class DirectorModelSelection:
    backend: str
    model_path: Path | None
    mmproj_path: Path | None


def director_model_files():
    models_root = Path(folder_paths.models_dir).resolve()
    files = []
    for relative_root in _MODEL_ROOTS:
        root = (models_root / relative_root).resolve()
        if not root.is_dir():
            continue
        for path in root.rglob("*.gguf"):
            resolved = path.resolve()
            if resolved.is_file() and resolved.is_relative_to(root):
                files.append(resolved.relative_to(models_root).as_posix())
    return sorted(set(files), key=str.casefold)


def director_model_options(projector=False):
    files = director_model_files()
    if projector:
        files = [path for path in files if "mmproj" in Path(path).name.casefold()]
    else:
        files = [path for path in files if "mmproj" not in Path(path).name.casefold()]
    return ["auto", *files]


def resolve_director_model(selection):
    if not selection or selection == "auto":
        return None
    relative = Path(selection)
    if relative.is_absolute() or ".." in relative.parts or "://" in str(selection):
        raise DirectorConfigurationError(f"Director model must be a registered relative path: {selection}")
    models_root = Path(folder_paths.models_dir).resolve()
    path = (models_root / relative).resolve()
    if not path.is_relative_to(models_root) or path.suffix.casefold() != ".gguf" or not path.is_file():
        raise DirectorConfigurationError(f"Invalid local director model: {selection}")
    if not any(path.is_relative_to((models_root / root).resolve()) for root in _MODEL_ROOTS):
        raise DirectorConfigurationError(f"Director model is outside the supported local model folders: {selection}")
    return path


def _qwen35_defaults():
    files = director_model_files()
    candidates = [path for path in files if "qwen3.5" in path.casefold()]
    models = [path for path in candidates if "mmproj" not in Path(path).name.casefold()]
    projectors = [path for path in candidates if "mmproj" in Path(path).name.casefold()]
    for model in models:
        projector = next((path for path in projectors if Path(path).parent == Path(model).parent), None)
        if projector is not None:
            return model, projector
    return None, None


def _qwen35_projector_for_model(model):
    parent = Path(model).parent
    return next((
        path for path in director_model_files()
        if Path(path).parent == parent and "mmproj" in Path(path).name.casefold()
    ), None)


def _qwen35_model_for_projector(mmproj):
    parent = Path(mmproj).parent
    return next((
        path for path in director_model_files()
        if Path(path).parent == parent and "mmproj" not in Path(path).name.casefold()
    ), None)


def resolve_director_selection(backend, model="auto", mmproj="auto"):
    if backend not in DIRECTOR_BACKENDS:
        raise DirectorConfigurationError(f"Unknown director backend: {backend}")
    if backend == "qwen3.5":
        if model == "auto" and mmproj == "auto":
            model, mmproj = _qwen35_defaults()
        elif model == "auto":
            model = _qwen35_model_for_projector(mmproj)
        elif mmproj == "auto":
            mmproj = _qwen35_projector_for_model(model)
    model_path = resolve_director_model(model)
    mmproj_path = resolve_director_model(mmproj)
    if model_path is not None and "mmproj" in model_path.name.casefold():
        raise DirectorConfigurationError("The director model selection cannot be an mmproj file")
    if mmproj_path is not None and "mmproj" not in mmproj_path.name.casefold():
        raise DirectorConfigurationError("The director mmproj selection must be an mmproj GGUF file")
    return DirectorModelSelection(backend, model_path, mmproj_path)
