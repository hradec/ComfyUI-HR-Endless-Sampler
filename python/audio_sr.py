"""AudioSR for ComfyUI AUDIO dictionaries, also usable as a WAV command-line tool.

Call fix_audio({"waveform": tensor_BCT, "sample_rate": 32000}) for 48 kHz audio.
Run with --install once to install the backend into ComfyUI's Python.
"""

import argparse
import importlib.util
import logging
import math
import os
import subprocess
import sys
import tempfile

import torch


class AudioSR:
    """Run isolated AudioSR inference while preserving channels and duration."""

    @staticmethod
    def validate(audio):
        """Validate the public AUDIO boundary before launching a worker."""
        if not isinstance(audio, dict):
            raise ValueError("AudioSR expects an AUDIO dictionary")
        waveform, rate = audio.get("waveform"), audio.get("sample_rate")
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or min(waveform.shape) < 1:
            raise ValueError("AudioSR waveform must have nonempty [batch, channels, samples] dimensions")
        if not waveform.is_floating_point() or not torch.isfinite(waveform).all():
            raise ValueError("AudioSR waveform must contain finite floating-point samples")
        if isinstance(rate, bool) or not isinstance(rate, int) or rate < 1 or rate > 384000:
            raise ValueError("AudioSR sample_rate must be a positive integer up to 384000")
        return waveform.detach().to(device="cpu", dtype=torch.float32), rate

    @classmethod
    def ensure_installed(cls):
        """Fail before an expensive render if this optional runtime is absent."""
        if not os.environ.get("HR_ENDLESS_AUDIOSR_PYTHON") and importlib.util.find_spec("audiosr") is None:
            raise RuntimeError("AudioSR is not installed. Run python/audio_sr.py --install with ComfyUI's Python, or disable audio_sr.")

    @classmethod
    def install(cls):
        """Install into this Python without downgrading ComfyUI's tested stack."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", os.path.join(root, "requirements.txt")], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "audiosr==0.0.7"], check=True)
        subprocess.run([sys.executable, os.path.abspath(__file__), "--check"], check=True)

    @classmethod
    def backend(cls):
        """Import the backend inside the disposable worker."""
        if importlib.util.find_spec("audiosr") is None:
            raise RuntimeError("AudioSR is not installed. Run python/audio_sr.py --install with ComfyUI's Python.")
        import audiosr
        return audiosr

    @classmethod
    def process(cls, audio, backend, model, seed=42, steps=50, guidance=3.5):
        """Low-pass at 12 kHz, then enhance overlapping windows without gain matching."""
        import soundfile
        import torchaudio
        from audiosr.lowpass import lowpass

        waveform, rate = cls.validate(audio)
        target_samples = max(1, round(waveform.shape[-1] * 48000 / rate))
        waveform = torchaudio.functional.resample(waveform, rate, 48000)[..., :target_samples]
        result = torch.zeros_like(waveform)
        # ponytail: AudioSR is mono. Independent channels preserve separation but
        # do not guarantee stereo phase coherence; a stereo-trained backend would.
        window_samples, overlap_samples = 245760, 30720  # 5.12 s with 0.64 s overlap.
        stride = window_samples - overlap_samples
        with tempfile.TemporaryDirectory(prefix="hr-audiosr-windows-") as directory:
            for batch in range(waveform.shape[0]):
                for channel in range(waveform.shape[1]):
                    source = waveform[batch, channel]
                    source_mean = source.mean()
                    if (source - source_mean).abs().max() <= 1e-7:
                        result[batch, channel] = source
                        continue
                    # AudioSR was trained with conventional low-pass degradation.
                    # Give it one fixed, clean cutoff for this evaluation.
                    filtered = torch.from_numpy(lowpass(source.numpy(), highcut=12000, fs=48000, order=8, _type="butter").copy()).to(dtype=torch.float32)
                    weights = torch.zeros(target_samples)
                    for start in range(0, target_samples, stride):
                        end = min(start + window_samples, target_samples)
                        original = filtered[start:end]
                        count = original.numel()
                        mean = original.mean()
                        peak = (original - mean).abs().max()
                        if peak <= 1e-7:
                            enhanced = original.clone()  # Preserve silence/DC; don't hallucinate noise.
                        else:
                            # Floating WAV avoids PCM quantization. Pad only for AudioSR's grid.
                            padded = torch.nn.functional.pad(original, (0, window_samples - count))
                            filename = os.path.join(directory, "input.wav")
                            soundfile.write(filename, padded.numpy(), 48000, subtype="FLOAT")
                            logging.info("AudioSR batch %d channel %d: %.3f-%.3f seconds", batch + 1, channel + 1, start / 48000.0, end / 48000.0)
                            generated = backend.super_resolution(model, filename, seed=(seed + start) % (2 ** 32), ddim_steps=steps, guidance_scale=guidance)
                            enhanced = torch.as_tensor(generated, dtype=torch.float32).reshape(-1)
                            if enhanced.numel() < count or not torch.isfinite(enhanced).all():
                                raise RuntimeError("AudioSR returned truncated or nonfinite audio")
                            enhanced = enhanced[:count].clone()

                        # Linear overlap-add uses only original inputs, never enhanced feedback.
                        weight = torch.ones(count)
                        fade = min(overlap_samples, count)
                        if start:
                            weight[:fade] = torch.linspace(0, 1, fade)
                        if end < target_samples:
                            weight[-fade:] = torch.linspace(1, 0, fade)
                        result[batch, channel, start:end] += enhanced * weight
                        weights[start:end] += weight
                        if end == target_samples:
                            break
                    if not torch.all(weights > 0):
                        raise RuntimeError("AudioSR window assembly left uncovered samples")
                    result[batch, channel] /= weights
        return {"waveform": result, "sample_rate": 48000}

    @classmethod
    def enhance(cls, audio, model_name="basic", device="cuda:0", seed=42, steps=50, guidance=3.5, interrupt_check=None):
        """Run one fresh worker; failures leave the caller's original audio untouched."""
        waveform, rate = cls.validate(audio)
        cls.ensure_installed()
        if model_name not in ("basic", "speech") or not isinstance(steps, int) or not 1 <= steps <= 1000:
            raise ValueError("AudioSR needs model basic/speech and 1-1000 sampling steps")
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("AudioSR guidance must be finite and nonnegative")
        log_handle, log_path = tempfile.mkstemp(prefix="hr-audiosr-", suffix=".log")
        os.close(log_handle)
        logging.info("AudioSR %s pass: %.3f seconds, %d channels; log: %s", model_name, waveform.shape[-1] / float(rate), waveform.shape[1], log_path)
        # ponytail: reload once per operation to release all GPU/RNG state. If
        # startup dominates, upgrade to a persistent worker with explicit offload.
        with tempfile.TemporaryDirectory(prefix="hr-audiosr-") as directory:
            request_path, output_path = os.path.join(directory, "request.pt"), os.path.join(directory, "output.pt")
            torch.save({"waveform": waveform, "sample_rate": rate}, request_path)
            command = [os.environ.get("HR_ENDLESS_AUDIOSR_PYTHON", sys.executable), os.path.abspath(__file__), "--worker", request_path, output_path, "--model", model_name, "--device", str(device), "--seed", str(int(seed) % (2 ** 32)), "--steps", str(steps), "--guidance", str(guidance)]
            with open(log_path, "w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while process.poll() is None:
                        if interrupt_check is not None:
                            interrupt_check()
                        try:
                            process.wait(timeout=0.25)
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                if process.returncode:
                    raise RuntimeError(f"AudioSR worker failed (exit {process.returncode}); see {log_path}. Run python/audio_sr.py --install if dependencies are missing.")
            enhanced = torch.load(output_path, map_location="cpu", weights_only=True)
            output, output_rate = cls.validate(enhanced)
            expected = (*waveform.shape[:2], max(1, round(waveform.shape[-1] * 48000 / rate)))
            if output_rate != 48000 or tuple(output.shape) != expected:
                raise RuntimeError(f"AudioSR changed channel count or duration; see {log_path}")
            return {"waveform": output, "sample_rate": output_rate}

    @classmethod
    def main(cls):
        """Provide installation, worker, and non-overwriting WAV CLI entry points."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("input", nargs="?")
        parser.add_argument("output", nargs="?")
        parser.add_argument("--install", action="store_true")
        parser.add_argument("--check", action="store_true")
        parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--model", choices=("basic", "speech"), default="basic")
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--steps", type=int, default=50)
        parser.add_argument("--guidance", type=float, default=3.5)
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        if args.install:
            cls.install()
            return
        if args.check:
            print("AudioSR backend:", cls.backend().__file__)
            return
        if not args.input or not args.output:
            parser.error("input and output paths are required")
        if os.path.exists(args.output):
            parser.error("output already exists; choose a new filename")
        if args.worker:
            # Only the published AudioSR checkpoint needs its legacy pickle loader;
            # internal request/output tensors always use weights_only=True.
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            backend = cls.backend()
            model = backend.build_model(model_name=args.model, device=args.device)
            audio = torch.load(args.input, map_location="cpu", weights_only=True)
            result = cls.process(audio, backend, model, seed=args.seed, steps=args.steps, guidance=args.guidance)
            torch.save(result, args.output)
        else:
            import soundfile
            waveform, rate = soundfile.read(args.input, dtype="float32", always_2d=True)
            audio = {"waveform": torch.from_numpy(waveform.T).unsqueeze(0), "sample_rate": int(rate)}
            result = cls.enhance(audio, model_name=args.model, device=args.device, seed=args.seed, steps=args.steps, guidance=args.guidance)
            # Exclusive creation prevents accidental replacement after inference.
            with open(args.output, "xb") as destination:
                soundfile.write(destination, result["waveform"][0].T.numpy(), result["sample_rate"], format="WAV", subtype="FLOAT")


def fix_audio(audio, **options):
    """Return a new 48 kHz AUDIO dict; the original decoded audio is never modified."""
    return AudioSR.enhance(audio, **options)


if __name__ == "__main__":
    AudioSR.main()
