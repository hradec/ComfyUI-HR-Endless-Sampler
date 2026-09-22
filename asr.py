"""Optional local faster-whisper adapter for reference media."""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import folder_paths
import torch
import torchaudio


def transcribe_audio(audio: Any, model_path: str, *, language: str = "") -> str:
    """Transcribe ComfyUI AUDIO data without importing another custom node."""
    if not model_path or not str(model_path).strip():
        return ""
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError("faster-whisper is required when whisper_model_path is set") from error
    relative = Path(str(model_path).strip())
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("whisper_model_path must be relative to the ComfyUI models directory")
    models_root = Path(folder_paths.models_dir).resolve()
    resolved = (models_root / relative).resolve()
    if not resolved.is_relative_to(models_root) or not resolved.is_dir():
        raise ValueError(f"Invalid local whisper model directory: {model_path}")
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("reference audio must be ComfyUI AUDIO data")
    waveform = audio["waveform"]
    sample_rate = int(audio.get("sample_rate") or 0)
    if sample_rate <= 0:
        raise ValueError("reference audio has an invalid sample_rate")
    if getattr(waveform, "ndim", 0) == 3:
        waveform = waveform[0]
    if getattr(waveform, "ndim", 0) != 2:
        raise ValueError("reference audio waveform must be [channels, samples]")
    samples = torch.nan_to_num(waveform.detach().to(device="cpu", dtype=torch.float32)).mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        samples = torchaudio.functional.resample(samples, sample_rate, 16000)
    model = WhisperModel(str(resolved), device="cpu", compute_type="int8", local_files_only=True)
    try:
        segments, _info = model.transcribe(samples[0].contiguous().numpy(), language=(language or None), vad_filter=True)
        return " ".join(str(segment.text).strip() for segment in segments if str(segment.text).strip())
    finally:
        del model
        gc.collect()


def transcribe_references(audios, model_path: str, *, language: str = "") -> list[str]:
    results = []
    for index, audio in enumerate(audios):
        try:
            results.append(transcribe_audio(audio, model_path, language=language))
        except Exception as error:
            logging.warning("Reference audio %d transcription failed: %s", index + 1, error)
            raise
    return results
