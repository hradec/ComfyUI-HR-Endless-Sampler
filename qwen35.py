"""Local Qwen3.5 multimodal director adapter."""

from pathlib import Path
from typing import Any

from .gemma4 import Gemma4ContinuityDirector, Gemma4ObservationError


class Qwen35ContinuityDirector(Gemma4ContinuityDirector):
    """Run the shared directing contract with a local Qwen3.5 GGUF pair."""

    def __init__(self, model_path: Path, mmproj_path: Path, debug=False,
                 capture_directory=None, observation_image_directory=None):
        super().__init__(
            debug=debug,
            gemma4_mtp=False,
            capture_directory=capture_directory,
            observation_image_directory=observation_image_directory,
        )
        self.model_path = Path(model_path).resolve()
        self.mmproj_path = Path(mmproj_path).resolve()

    def _configure_request(self, request: dict[str, Any]) -> None:
        request["debug"] = self.debug
        request["gemma4_mtp"] = False
        request["director_backend"] = "qwen3.5"
        request["director_model_path"] = str(self.model_path)
        request["director_mmproj_path"] = str(self.mmproj_path)
        request["director_n_ctx"] = 4096
        request["director_n_batch"] = 256

    def materialize_preproduction_cache(self, request, timing_plan, progress_callback=None):
        raise Gemma4ObservationError("Qwen3.5 director does not support the Gemma preproduction KV cache")
