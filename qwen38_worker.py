"""Dedicated direct-process entrypoint for HR Endless Qwen3.6/Qwen3.8.

Qwen3.5 continues to execute qwen35.py. The newer Qwen3.6/Qwen3.8 families
share qwen36_38.py and never import the Qwen3.5 runtime module.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


_ROOT = Path(__file__).resolve().parent
_PACKAGE = "hr_endless_qwen38_worker"


def _load_package_module(name):
    package = sys.modules.get(_PACKAGE)
    if package is None:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [str(_ROOT)]
        sys.modules[_PACKAGE] = package
    qualified = f"{_PACKAGE}.{name}"
    existing = sys.modules.get(qualified)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(qualified, _ROOT / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name}.py for Qwen3.8 worker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


def main():
    _load_package_module("director_errors")
    _load_package_module("story_format")
    runtime = _load_package_module("qwen36_38")
    return runtime._worker_main()


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("qwen38_worker.py is an internal Qwen3.6/3.8 worker; use it through the sampler node")
    raise SystemExit(main())
