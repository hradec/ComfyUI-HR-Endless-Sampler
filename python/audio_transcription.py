"""Independent recognition of generated H3 speech for chunk dialogue continuity."""

import numpy as np
import librosa
import torch
import whisper


class ChunkAudioTranscriber:
    """Load one multilingual Whisper model and transcribe decoded audio chunks."""

    def __init__(self, model_name="small"):
        """Keep model loading lazy so workflows without teacher audio pay nothing."""
        self.model_name = model_name
        self.model = None

    def transcribe(self, waveform, sample_rate):
        """Return independently recognized text and timed words from a PCM tensor."""
        if sample_rate <= 0 or waveform.ndim not in (2, 3) or waveform.shape[-1] == 0:
            raise ValueError("Audio transcription needs nonempty [batch, channels, samples] PCM and a positive sample rate")
        audio = waveform.detach().to(device="cpu", dtype=torch.float32)
        if audio.ndim == 3:
            audio = audio.mean(dim=(0, 1))
        else:
            audio = audio.mean(dim=0)
        samples = audio.numpy()
        if sample_rate != 16000:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        samples = np.asarray(samples, dtype=np.float32)
        if self.model is None:
            self.model = whisper.load_model(self.model_name, device="cpu")
        result = self.model.transcribe(samples, task="transcribe", fp16=False, verbose=False, word_timestamps=True, condition_on_previous_text=False)
        words = [{"word": word["word"].strip(), "start": float(word["start"]), "end": float(word["end"]), "probability": float(word.get("probability", 0))} for segment in result.get("segments", ()) for word in segment.get("words", ())]
        return {"text": str(result.get("text", "")).strip(), "words": words, "language": result.get("language")}
