"""Run: ComfyUI's python.sh -m unittest discover -s python -p test_audio_sr.py."""

import unittest
from unittest.mock import patch

import soundfile
import torch
import torchaudio

from audio_sr import AudioSR, fix_audio


class IdentityBackend:
    """Return a distinct raw level while recording every filtered window."""

    def __init__(self):
        """Track window inputs without loading any neural network."""
        self.inputs = []

    def super_resolution(self, _model, filename, **options):
        """Double the filtered input so any output gain matching breaks the test."""
        samples, rate = soundfile.read(filename, dtype="float32")
        self.inputs.append((samples.copy(), rate, options["seed"]))
        return (torch.from_numpy(samples) * 2).reshape(1, 1, -1)


class AudioSRTest(unittest.TestCase):
    """Check exact duration, original preservation, worker boundaries, and seams."""

    def test_short_and_long_stereo_use_filtered_raw_output_and_preserve_input(self):
        """Windows return raw backend audio after the fixed 12 kHz low-pass."""
        from audiosr.lowpass import lowpass

        for seconds in (0.125, 5.13, 11.0):
            rate = 32000
            time = torch.arange(round(seconds * rate)) / rate
            waveform = torch.stack((0.1 * torch.sin(2 * torch.pi * 100 * time), 0.3 * torch.cos(2 * torch.pi * 200 * time))).unsqueeze(0)
            original = waveform.clone()
            backend = IdentityBackend()
            result = AudioSR.process({"waveform": waveform, "sample_rate": rate}, backend, None)
            resampled = torchaudio.functional.resample(original, rate, 48000)
            expected = torch.stack(tuple(torch.from_numpy(lowpass(channel.numpy(), highcut=12000, fs=48000, order=8, _type="butter").copy()) for channel in resampled[0])).unsqueeze(0).to(dtype=torch.float32) * 2
            self.assertEqual(result["sample_rate"], 48000)
            self.assertEqual(tuple(result["waveform"].shape), (1, 2, round(seconds * 48000)))
            self.assertTrue(torch.allclose(result["waveform"], expected, atol=1e-6))
            self.assertTrue(torch.equal(waveform, original))
            self.assertTrue(all(rate == 48000 and len(samples) == 245760 for samples, rate, _seed in backend.inputs))
            self.assertGreaterEqual(len(backend.inputs), 2)

    def test_silence_and_dc_do_not_generate_noise(self):
        """Silent or constant channels bypass model inference."""
        waveform = torch.tensor([0.0, 0.15]).reshape(1, 2, 1).expand(1, 2, 4800).clone()
        backend = IdentityBackend()
        result = AudioSR.process({"waveform": waveform, "sample_rate": 48000}, backend, None)
        self.assertTrue(torch.equal(result["waveform"], waveform))
        self.assertEqual(backend.inputs, [])

    def test_invalid_audio_is_rejected(self):
        """Malformed inputs must fail before worker creation."""
        for audio in ({}, {"waveform": torch.zeros(2, 100), "sample_rate": 32000}, {"waveform": torch.ones(1, 1, 10), "sample_rate": 0}, {"waveform": torch.full((1, 1, 10), float("nan")), "sample_rate": 32000}):
            with self.assertRaises(ValueError):
                AudioSR.validate(audio)

    def test_worker_roundtrip_and_input_immutability(self):
        """The simple function returns validated new audio, never mutating its input."""
        class Worker:
            """Represent a completed worker with an independently produced output."""

            returncode = 0

            def __init__(self, command, **_options):
                """Write the result through the same tensor-file protocol."""
                original = torch.load(command[3], weights_only=True)
                wave = torchaudio.functional.resample(original["waveform"], original["sample_rate"], 48000)
                torch.save({"waveform": wave * 2, "sample_rate": 48000}, command[4])

            def poll(self):
                """Report successful completion."""
                return self.returncode

        original = {"waveform": torch.ones(1, 2, 320), "sample_rate": 32000}
        with patch.object(AudioSR, "ensure_installed"), patch("audio_sr.subprocess.Popen", Worker):
            result = fix_audio(original)
        self.assertTrue(torch.equal(original["waveform"], torch.ones(1, 2, 320)))
        self.assertEqual(tuple(result["waveform"].shape), (1, 2, 480))

    def test_interrupt_terminates_the_worker(self):
        """Cancellation must not leave an AudioSR model resident on the GPU."""
        class Worker:
            """Stay alive until the caller terminates this process."""

            returncode = None

            def poll(self):
                """Report whether termination occurred."""
                return self.returncode

            def terminate(self):
                """Record process termination."""
                self.returncode = -15

            def wait(self, timeout=None):
                """Return the recorded exit status."""
                return self.returncode

        def cancel():
            """Simulate ComfyUI cancellation without importing its server."""
            raise InterruptedError("cancelled")

        worker = Worker()
        with patch.object(AudioSR, "ensure_installed"), patch("audio_sr.subprocess.Popen", return_value=worker), self.assertRaises(InterruptedError):
            fix_audio({"waveform": torch.ones(1, 1, 100), "sample_rate": 48000}, interrupt_check=cancel)
        self.assertEqual(worker.returncode, -15)


if __name__ == "__main__":
    unittest.main()
