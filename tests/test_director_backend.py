import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
spec = importlib.util.spec_from_file_location("hr_endless_director_backend_test", PLUGIN_ROOT / "director_backend.py")
director_backend = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = director_backend
spec.loader.exec_module(director_backend)


class DirectorBackendTest(unittest.TestCase):
    def test_ui_exposes_each_qwen_backend(self):
        self.assertEqual(
            director_backend.DIRECTOR_BACKENDS,
            ("gemma4", "qwen3.5", "qwen3.6", "qwen3.8"),
        )

    def test_qwen_auto_selects_local_model_and_projector(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary)
            directory = models / "LLM" / "GGUF" / "qwen3.5-9B"
            directory.mkdir(parents=True)
            model = directory / "Qwen3.5-9B.Q4_K_M.gguf"
            mmproj = directory / "mmproj-Qwen3.5-9B.gguf"
            model.touch()
            mmproj.touch()
            with patch.object(director_backend.folder_paths, "models_dir", str(models)):
                selection = director_backend.resolve_director_selection("qwen3.5")
        self.assertEqual(selection.model_path, model.resolve())
        self.assertEqual(selection.mmproj_path, mmproj.resolve())

    def test_rejects_paths_outside_registered_model_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary) / "models"
            models.mkdir()
            outside = Path(temporary) / "outside.gguf"
            outside.touch()
            with patch.object(director_backend.folder_paths, "models_dir", str(models)):
                with self.assertRaises(director_backend.DirectorConfigurationError):
                    director_backend.resolve_director_model(str(outside))

    def test_qwen38_auto_selects_only_qwen38_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary)
            for family in ("qwen3.5", "qwen3.8"):
                directory = models / "LLM" / "GGUF" / family
                directory.mkdir(parents=True)
                (directory / f"{family}-model.gguf").touch()
                (directory / f"mmproj-{family}.gguf").touch()
            with patch.object(director_backend.folder_paths, "models_dir", str(models)):
                selection = director_backend.resolve_director_selection("qwen3.8")
        self.assertIn("qwen3.8", selection.model_path.as_posix())
        self.assertIn("qwen3.8", selection.mmproj_path.as_posix())

    def test_qwen_backend_rejects_another_qwen_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary)
            directory = models / "LLM" / "GGUF" / "qwen3.8"
            directory.mkdir(parents=True)
            model = directory / "qwen3.8-model.gguf"
            projector = directory / "mmproj-qwen3.8.gguf"
            model.touch()
            projector.touch()
            with patch.object(director_backend.folder_paths, "models_dir", str(models)):
                with self.assertRaisesRegex(director_backend.DirectorConfigurationError, "qwen3.6 backend"):
                    director_backend.resolve_director_selection(
                        "qwen3.6", model.relative_to(models).as_posix(), projector.relative_to(models).as_posix(),
                    )

    def test_qwen_auto_does_not_mix_model_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary)
            first = models / "LLM" / "GGUF" / "qwen3.5-a"
            second = models / "LLM" / "GGUF" / "qwen3.5-b"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "model-a.gguf").touch()
            (second / "model-b.gguf").touch()
            (second / "mmproj-b.gguf").touch()
            with patch.object(director_backend.folder_paths, "models_dir", str(models)):
                selection = director_backend.resolve_director_selection("qwen3.5")
        self.assertEqual(selection.model_path.parent, second.resolve())
        self.assertEqual(selection.mmproj_path.parent, second.resolve())

    def test_qwen_explicit_model_auto_projector_uses_same_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary)
            first = models / "LLM" / "GGUF" / "qwen3.5-a"
            second = models / "LLM" / "GGUF" / "qwen3.5-b"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "model-a.gguf").touch()
            (first / "mmproj-a.gguf").touch()
            model = second / "model-b.gguf"
            projector = second / "mmproj-b.gguf"
            model.touch()
            projector.touch()
            with patch.object(director_backend.folder_paths, "models_dir", str(models)):
                selection = director_backend.resolve_director_selection(
                    "qwen3.5",
                    model.relative_to(models).as_posix(),
                    "auto",
                )
        self.assertEqual(selection.model_path, model.resolve())
        self.assertEqual(selection.mmproj_path, projector.resolve())

    def test_rejects_absolute_path_inside_models_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary)
            directory = models / "LLM" / "GGUF"
            directory.mkdir(parents=True)
            model = directory / "model.gguf"
            model.touch()
            with patch.object(director_backend.folder_paths, "models_dir", str(models)):
                with self.assertRaises(director_backend.DirectorConfigurationError):
                    director_backend.resolve_director_model(str(model.resolve()))

    def test_rejects_model_projector_role_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary)
            directory = models / "LLM" / "GGUF"
            directory.mkdir(parents=True)
            model = directory / "model.gguf"
            mmproj = directory / "mmproj-model.gguf"
            model.touch()
            mmproj.touch()
            with patch.object(director_backend.folder_paths, "models_dir", str(models)):
                with self.assertRaises(director_backend.DirectorConfigurationError):
                    director_backend.resolve_director_selection("qwen3.5", mmproj.relative_to(models).as_posix(), model.relative_to(models).as_posix())

    def test_rejects_explicit_qwen_model_and_projector_from_different_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary)
            model_directory = models / "LLM" / "GGUF" / "qwen3.8"
            projector_directory = models / "LLM" / "GGUF" / "qwen3.5"
            model_directory.mkdir(parents=True)
            projector_directory.mkdir(parents=True)
            model = model_directory / "Qwen3.8.gguf"
            projector = projector_directory / "mmproj-Qwen3.5.gguf"
            model.touch()
            projector.touch()
            with patch.object(director_backend.folder_paths, "models_dir", str(models)):
                with self.assertRaisesRegex(director_backend.DirectorConfigurationError, "same model directory"):
                    director_backend.resolve_director_selection(
                        "qwen3.5",
                        model.relative_to(models).as_posix(),
                        projector.relative_to(models).as_posix(),
                    )

    def test_qwen_adapter_configures_local_non_mtp_worker(self):
        package_name = "hr_endless_qwen_adapter_test"
        package = types.ModuleType(package_name)
        package.__path__ = [str(PLUGIN_ROOT)]
        fake_gemma = types.ModuleType(package_name + ".gemma4")

        class FakeDirector:
            def __init__(self, debug=False, **_kwargs):
                self.debug = debug

        fake_gemma.Gemma4ContinuityDirector = FakeDirector
        fake_gemma.Gemma4ObservationError = RuntimeError
        spec = importlib.util.spec_from_file_location(
            package_name + ".qwen35",
            PLUGIN_ROOT / "qwen35.py",
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            package_name: package,
            package_name + ".gemma4": fake_gemma,
            spec.name: module,
        }):
            spec.loader.exec_module(module)
            director = module.Qwen35ContinuityDirector(Path("model.gguf"), Path("mmproj.gguf"), debug=True)
            request = {}
            director._configure_request(request)

        self.assertEqual(request["director_backend"], "qwen3.5")
        self.assertFalse(request["gemma4_mtp"])
        self.assertEqual(request["director_n_ctx"], 16384)
        self.assertEqual(request["director_n_batch"], 256)
        self.assertTrue(request["director_model_path"].endswith("model.gguf"))
        self.assertTrue(request["director_mmproj_path"].endswith("mmproj.gguf"))


if __name__ == "__main__":
    unittest.main()
