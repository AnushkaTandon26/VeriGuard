#!/usr/bin/env python3
"""Local web app for FA3-CLIP deepfake-image classification.

Run with: ./venv/bin/python app.py
Then open: http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from audio_inference import AudioPredictionError, CNNConformerPredictor
from inference import FA3Predictor, PredictionError


ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "web"
SAMPLE_DIR = ROOT_DIR / "datasets" / "dffmd_images_sample" / "test"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave", "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/webm", "audio/mp4", "audio/x-m4a", "audio/flac",
}
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".ogg", ".webm", ".m4a", ".mp4", ".flac"}


class DeepfakeHandler(BaseHTTPRequestHandler):
    predictor: FA3Predictor
    audio_predictor: CNNConformerPredictor

    def log_message(self, format, *args):  # Keep request logs compact and useful.
        print(f"[{self.log_date_time_string()}] {format % args}")

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, allowed_root: Path = STATIC_DIR) -> None:
        if not path.is_file() or allowed_root not in path.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(HTTPStatus.OK, {"image": self.predictor.status(), "audio": self.audio_predictor.status()})
            return
        if path == "/api/samples":
            self._send_json(HTTPStatus.OK, {"samples": self._demo_samples()})
            return
        if path.startswith("/samples/"):
            relative_path = Path(unquote(path.removeprefix("/samples/")))
            self._send_file((SAMPLE_DIR / relative_path).resolve(), SAMPLE_DIR.resolve())
            return
        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html")
            return
        if path in ("/app.js", "/styles.css"):
            self._send_file(STATIC_DIR / path.lstrip("/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    @staticmethod
    def _demo_samples() -> list[dict]:
        """Expose only the small, bundled test examples for the UI demo cards."""
        samples = []
        for label in ("real", "fake"):
            for image_path in sorted((SAMPLE_DIR / label).glob("*"))[:2]:
                if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                relative_path = image_path.relative_to(SAMPLE_DIR).as_posix()
                samples.append(
                    {
                        "name": image_path.name,
                        "label": label,
                        "url": f"/samples/{quote(relative_path)}",
                    }
                )
        return samples

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/predict", "/api/predict-audio"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if not 0 < content_length <= MAX_UPLOAD_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Upload a file smaller than 12 MB."})
            return

        try:
            field_name = "image" if path == "/api/predict" else "audio"
            filename, content_type, file_bytes = self._read_upload(content_length, field_name)
            if path == "/api/predict":
                if content_type not in ALLOWED_IMAGE_TYPES:
                    raise ValueError("Use a JPG, PNG, or WebP image.")
                result = self.predictor.predict(file_bytes)
            else:
                suffix = Path(filename).suffix.lower()
                if content_type not in ALLOWED_AUDIO_TYPES and suffix not in ALLOWED_AUDIO_SUFFIXES:
                    raise ValueError("Use WAV, MP3, OGG, WebM, M4A, or FLAC audio.")
                result = self.audio_predictor.predict(file_bytes, filename)
        except ValueError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except (PredictionError, AudioPredictionError) as error:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
        except Exception:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "The file could not be analysed. Please try another file."})
        else:
            result.update({"filename": filename, "content_type": content_type})
            self._send_json(HTTPStatus.OK, result)

    def _read_upload(self, content_length: int, expected_field: str) -> tuple[str, str, bytes]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Send the image as multipart/form-data.")
        raw_body = self.rfile.read(content_length)
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + raw_body
        )
        for part in message.iter_parts():
            if part.get_param("name", header="content-disposition") != expected_field:
                continue
            image_bytes = part.get_payload(decode=True) or b""
            image_type = part.get_content_type()
            filename = Path(part.get_filename() or "uploaded-image").name
            if not image_bytes:
                raise ValueError(f"The uploaded {expected_field} file is empty.")
            return filename, image_type, image_bytes
        raise ValueError(f"Choose an {expected_field} file before analysing.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local FA3-CLIP web interface.")
    parser.add_argument("--host", default="127.0.0.1", help="Server address (default: 127.0.0.1).")
    parser.add_argument("--port", default=8000, type=int, help="Server port (default: 8000).")
    parser.add_argument("--model-dir", default=str(ROOT_DIR / "models" / "fa3_dffmd"), help="FA3 checkpoint directory.")
    parser.add_argument("--audio-model", default=str(ROOT_DIR / "models" / "audio" / "cnn_conformer.h5"), help="CNN-Conformer audio checkpoint.")
    args = parser.parse_args()

    DeepfakeHandler.predictor = FA3Predictor(Path(args.model_dir))
    DeepfakeHandler.audio_predictor = CNNConformerPredictor(Path(args.audio_model))
    server = ThreadingHTTPServer((args.host, args.port), DeepfakeHandler)
    print(f"FA3 Deepfake Detector is ready at http://{args.host}:{args.port}")
    print("The trained model loads automatically on the first analysis.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
