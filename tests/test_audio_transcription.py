"""Small checks for generated-audio transcription without model downloads."""

import importlib
import os
import sys
import unittest
from unittest.mock import Mock, patch

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(ROOT)))
sys.path.insert(0, os.path.dirname(ROOT))
module = importlib.import_module(os.path.basename(ROOT) + ".python.audio_transcription")


class AudioTranscriptionTest(unittest.TestCase):
    """Verify that ASR sees decoded audio without the expected dialogue as a hint."""

    def test_transcribes_mono_audio_once_and_reuses_model(self):
        """Every request is independent, timed, and uses the same loaded model."""
        model = Mock()
        model.transcribe.return_value = {"text": " hello world ", "language": "en", "segments": [{"words": [{"word": " hello", "start": 0.1, "end": 0.3, "probability": 0.9}]}]}
        with patch.object(module.whisper, "load_model", return_value=model) as loader:
            transcriber = module.ChunkAudioTranscriber()
            waveform = torch.ones((1, 2, 16000))
            first = transcriber.transcribe(waveform, 16000)
            second = transcriber.transcribe(waveform, 16000)
        self.assertEqual(first, second)
        self.assertEqual(first["text"], "hello world")
        self.assertEqual(first["words"][0]["end"], 0.3)
        loader.assert_called_once_with("small", device="cpu")
        self.assertEqual(model.transcribe.call_count, 2)
        self.assertNotIn("initial_prompt", model.transcribe.call_args.kwargs)
        with self.assertRaises(ValueError):
            transcriber.transcribe(torch.empty((1, 1, 0)), 16000)

    def test_resamples_decoded_multichannel_audio_to_whisper_rate(self):
        """VAE output reaches Whisper at 16 kHz without stereo duplication."""
        model = Mock()
        model.transcribe.return_value = {"text": "", "segments": []}
        with patch.object(module.whisper, "load_model", return_value=model), patch.object(module.librosa, "resample", return_value=module.np.zeros(16000, dtype=module.np.float32)) as resample:
            module.ChunkAudioTranscriber().transcribe(torch.ones((1, 2, 32000)), 32000)
        self.assertEqual(resample.call_args.kwargs, {"orig_sr": 32000, "target_sr": 16000})
        self.assertEqual(resample.call_args.args[0].shape, (32000,))
        self.assertEqual(model.transcribe.call_args.args[0].shape, (16000,))
