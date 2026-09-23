"""CNN--Conformer audio deepfake inference service.

The feature extraction in this module matches the training/inference code that
ships with ``models/audio/cnn_conformer.h5``: a 16 kHz, 128-bin log-mel
spectrogram, cropped or padded to 128 frames.
"""

from __future__ import annotations

import threading
import time
import tempfile
from importlib.util import find_spec
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np


class AudioPredictionError(RuntimeError):
    """A recoverable audio-model or audio-decoding failure."""


class CNNConformerPredictor:
    sample_rate = 16_000
    input_shape = (128, 128, 1)

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def status(self) -> dict:
        available, detail = self._dependencies_available()
        return {
            "model": "CNN-Conformer (audio)",
            "model_loaded": self.model is not None,
            "checkpoint": str(self.model_path),
            "dependencies_ready": available,
            "detail": detail,
        }

    @staticmethod
    def _dependencies_available() -> tuple[bool, str]:
        # Do not import TensorFlow just to answer /api/health: importing it
        # takes several seconds on CPU-only machines.
        missing = [name for name in ("librosa", "soundfile", "tensorflow") if find_spec(name) is None]
        if missing:
            return False, f"Missing audio dependency: {', '.join(missing)}"
        return True, "ready"

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            if not self.model_path.is_file():
                raise AudioPredictionError(f"CNN-Conformer model checkpoint not found: {self.model_path}")
            ready, detail = self._dependencies_available()
            if not ready:
                raise AudioPredictionError(f"Audio analysis is not available. {detail}. Run: ./venv/bin/pip install -r requirements.txt")
            try:
                import tensorflow as tf

                self.model = tf.keras.models.load_model(self.model_path, compile=False)
            except Exception as error:
                raise AudioPredictionError(f"Unable to load the CNN-Conformer model: {error}") from error

    def _decode_audio(self, audio_bytes: bytes, filename: str) -> tuple[np.ndarray, int]:
        if not audio_bytes:
            raise ValueError("The uploaded audio is empty.")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".wav", ".mp3", ".ogg", ".webm", ".m4a", ".mp4", ".flac"}:
            suffix = ".webm"
        temporary_path = None
        try:
            import librosa

            # A temporary file preserves the upload's codec suffix, allowing
            # librosa/audioread to decode browser-recorded WebM as well as WAV.
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
                temporary_file.write(audio_bytes)
                temporary_path = temporary_file.name
            samples, sample_rate = librosa.load(temporary_path, sr=self.sample_rate, mono=True)
        except Exception as error:
            raise ValueError("The audio could not be decoded. Use WAV, MP3, OGG, M4A, or WebM audio.") from error
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)
        if samples.size < self.sample_rate // 2:
            raise ValueError("Use an audio clip that is at least 0.5 seconds long.")
        if not np.isfinite(samples).all():
            raise ValueError("The audio contains invalid samples.")
        return samples.astype(np.float32), sample_rate

    def _preprocess(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        import librosa
        from scipy.signal import butter, filtfilt

        if sample_rate != self.sample_rate:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=self.sample_rate)
        nyquist = self.sample_rate / 2
        b, a = butter(4, [80 / nyquist, min(8000 / nyquist, 0.999)], btype="band")
        try:
            samples = filtfilt(b, a, samples).astype(np.float32)
        except ValueError as error:
            raise ValueError("The audio clip is too short to analyse.") from error
        peak = float(np.max(np.abs(samples)))
        if peak > 0:
            samples = samples / peak
        mel = librosa.feature.melspectrogram(
            y=samples, sr=self.sample_rate, n_fft=2048, hop_length=256,
            n_mels=128, fmin=80, fmax=8000, power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        if log_mel.shape[1] < 128:
            log_mel = np.pad(log_mel, ((0, 0), (0, 128 - log_mel.shape[1])), mode="constant")
        else:
            log_mel = log_mel[:, :128]
        normalized = np.clip((log_mel + 80) / 80, 0, 1)
        return normalized[np.newaxis, ..., np.newaxis].astype(np.float32)

    def predict(self, audio_bytes: bytes, filename: str = "uploaded-audio") -> dict:
        samples, sample_rate = self._decode_audio(audio_bytes, filename)
        features = self._preprocess(samples, sample_rate)
        self._ensure_loaded()
        started_at = time.perf_counter()
        with self._predict_lock:
            raw_score = float(np.squeeze(self.model.predict(features, verbose=0)))
        inference_ms = round((time.perf_counter() - started_at) * 1000)
        fake_probability = float(np.clip(raw_score, 0.0, 1.0))
        real_probability = 1.0 - fake_probability
        label = "fake" if fake_probability >= 0.5 else "real"
        confidence = max(fake_probability, real_probability)
        duration = round(samples.size / sample_rate, 2)
        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {"fake": round(fake_probability * 100, 2), "real": round(real_probability * 100, 2)},
            "model": "CNN-Conformer",
            "audio": {"duration_seconds": duration, "sample_rate": self.sample_rate, "features": "128-bin log-mel spectrogram"},
            "inference_ms": inference_ms,
            "assessment": "high" if confidence >= 0.85 else "moderate" if confidence >= 0.65 else "low",
            "scan_id": f"AUDIO-{uuid4().hex[:8].upper()}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
