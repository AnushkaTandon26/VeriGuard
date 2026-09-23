"""Reusable, lazy FA3-DFFMD inference service."""

from __future__ import annotations

import io
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from PIL import Image, UnidentifiedImageError

from model import load_model


class PredictionError(RuntimeError):
    """A recoverable failure while preparing the model or image."""


class FA3Predictor:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def status(self) -> dict:
        return {
            "status": "ready",
            "model": "FA3-CLIP (DFFMD)",
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "checkpoint": str(self.model_dir),
        }

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            if not (self.model_dir / "fa3_model.pt").is_file():
                raise PredictionError(f"FA3 model checkpoint not found: {self.model_dir / 'fa3_model.pt'}")
            try:
                model, processor = load_model(self.model_dir)
                self.model = model.to(self.device).eval()
                self.processor = processor
            except Exception as error:
                raise PredictionError(f"Unable to load the FA3 model: {error}") from error

    def predict(self, image_bytes: bytes) -> dict:
        try:
            uploaded_image = Image.open(io.BytesIO(image_bytes))
            metadata = {
                "width": uploaded_image.width,
                "height": uploaded_image.height,
                "format": uploaded_image.format or "Image",
            }
            image = uploaded_image.convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("The upload is not a valid image.") from error

        self._ensure_loaded()
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        started_at = time.perf_counter()
        with self._predict_lock, torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1).detach().cpu()
        inference_ms = round((time.perf_counter() - started_at) * 1000)

        fake_probability = float(probabilities[0].item())
        real_probability = float(probabilities[1].item())
        label = "fake" if fake_probability >= real_probability else "real"
        confidence = max(fake_probability, real_probability)
        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {"fake": round(fake_probability * 100, 2), "real": round(real_probability * 100, 2)},
            "model": "FA3-CLIP (DFFMD)",
            "image": metadata,
            "inference_ms": inference_ms,
            "assessment": "high" if confidence >= 0.85 else "moderate" if confidence >= 0.65 else "low",
            "scan_id": f"FA3-{uuid4().hex[:8].upper()}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
