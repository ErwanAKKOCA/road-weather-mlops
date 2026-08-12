from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import Settings


class InvalidImageError(ValueError):
    """Raised when an uploaded image cannot be decoded safely."""


class Engine(Protocol):
    name: str
    region: str

    def predict(self, rgb: Image.Image, semantic_mask: Image.Image) -> np.ndarray: ...


@dataclass(frozen=True)
class ServicePrediction:
    sequence_id: str
    frame_index: int
    predicted_class: str
    confidence: float
    entropy: float
    abstained: bool
    raw_probabilities: np.ndarray
    temporal_probabilities: np.ndarray
    inference_ms: float


@dataclass
class _SequenceState:
    probabilities: np.ndarray
    frame_index: int
    updated_at: float


class EMASequenceStore:
    """Thread-safe, bounded, in-memory causal EMA state."""

    def __init__(self, alpha: float, ttl_seconds: int, max_sequences: int):
        self.alpha = alpha
        self.ttl_seconds = ttl_seconds
        self.max_sequences = max_sequences
        self._states: OrderedDict[str, _SequenceState] = OrderedDict()
        self._lock = threading.Lock()

    def update(
        self, sequence_id: str, raw: np.ndarray, reset: bool = False
    ) -> tuple[np.ndarray, int]:
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            if reset:
                self._states.pop(sequence_id, None)
            previous = self._states.pop(sequence_id, None)
            if previous is None:
                temporal = raw.copy()
                frame_index = 0
            else:
                temporal = self.alpha * raw + (1.0 - self.alpha) * previous.probabilities
                temporal = temporal / np.clip(temporal.sum(), 1e-12, None)
                frame_index = previous.frame_index + 1
            self._states[sequence_id] = _SequenceState(temporal, frame_index, now)
            while len(self._states) > self.max_sequences:
                self._states.popitem(last=False)
            return temporal.copy(), frame_index

    def reset(self, sequence_id: str) -> bool:
        with self._lock:
            return self._states.pop(sequence_id, None) is not None

    def active_sequence_count(self) -> int:
        with self._lock:
            self._evict(time.monotonic())
            return len(self._states)

    def _evict(self, now: float) -> None:
        expired = [
            key for key, state in self._states.items() if now - state.updated_at > self.ttl_seconds
        ]
        for key in expired:
            self._states.pop(key, None)


def decode_image(payload: bytes, max_bytes: int, label: str) -> Image.Image:
    if not payload:
        raise InvalidImageError(f"{label} is empty.")
    if len(payload) > max_bytes:
        raise InvalidImageError(f"{label} exceeds the {max_bytes}-byte limit.")
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"{label} is not a valid image.") from exc


class MockEngine:
    """Deterministic smoke-test engine; never presented as scientific output."""

    name = "mock_smoke_test"
    region = "semantic_sky"

    def predict(self, rgb: Image.Image, semantic_mask: Image.Image) -> np.ndarray:
        values = np.asarray(rgb.resize((32, 32)), dtype=np.float64).mean(axis=(0, 1)) / 255.0
        logits = np.array([values.mean(), 1.0 - values[2], values[2]]) * 3.0
        logits -= logits.max()
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum()


class DINOv2SkyEngine:
    name = "dinov2_logistic"
    region = "semantic_sky"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_path = settings.artifact_dir / settings.model_filename
        self.region_path = settings.artifact_dir / settings.region_filename
        self._verify_release()
        self.region_definition = json.loads(self.region_path.read_text(encoding="utf-8"))
        self.classifier = joblib.load(self.model_path)

        import timm
        import torch
        from timm.data import create_transform, resolve_model_data_config

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = (
            timm.create_model(settings.dino_backbone, pretrained=True, num_classes=0)
            .eval()
            .to(self.device)
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        data_config = resolve_model_data_config(self.backbone)
        self.transform = create_transform(**data_config, is_training=False)

    def _verify_release(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Missing Phase VI model artifact: {self.model_path}")
        if not self.region_path.is_file():
            raise FileNotFoundError(f"Missing Phase VI region definition: {self.region_path}")
        manifest_path = self.settings.artifact_dir / self.settings.release_manifest_filename
        if not manifest_path.is_file():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = {entry["filename"]: entry for entry in manifest.get("files", [])}
        for path in (self.model_path, self.region_path):
            expected = entries.get(path.name, {}).get("sha256")
            if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"SHA-256 mismatch for {path.name}")

    def _render_sky(self, rgb: Image.Image, semantic_mask: Image.Image) -> Image.Image:
        if rgb.size != semantic_mask.size:
            raise InvalidImageError("RGB and semantic-mask dimensions must match.")
        rgb_array = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
        mask_array = np.asarray(semantic_mask.convert("RGB"), dtype=np.uint8)
        region_classes = self.region_definition["regions"]["semantic_sky"]
        color_map = self.region_definition["class_colors_rgb"]
        selected = np.zeros(mask_array.shape[:2], dtype=bool)
        for class_name in region_classes:
            color = np.asarray(color_map[class_name], dtype=np.uint8)
            selected |= np.all(mask_array == color, axis=-1)
        if not selected.any():
            raise InvalidImageError("The semantic mask contains no configured sky pixels.")
        rendered = np.full_like(rgb_array, self.settings.neutral_background)
        rendered[selected] = rgb_array[selected]
        return Image.fromarray(rendered)

    def predict(self, rgb: Image.Image, semantic_mask: Image.Image) -> np.ndarray:
        regional_image = self._render_sky(rgb, semantic_mask)
        tensor = self.transform(regional_image).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            embedding = self.backbone(tensor)
        embedding = embedding.detach().cpu().numpy().reshape(1, -1)
        raw = np.asarray(self.classifier.predict_proba(embedding)[0], dtype=np.float64)
        aligned = np.zeros(len(self.settings.classes), dtype=np.float64)
        classes = self.classifier.named_steps["classifier"].classes_
        for source_index, class_id in enumerate(classes):
            aligned[int(class_id)] = raw[source_index]
        return aligned / np.clip(aligned.sum(), 1e-12, None)


class WeatherService:
    def __init__(self, settings: Settings, engine: Engine):
        self.settings = settings
        self.engine = engine
        self.store = EMASequenceStore(
            settings.ema_alpha, settings.sequence_ttl_seconds, settings.max_sequences
        )

    def predict(
        self,
        sequence_id: str,
        rgb_bytes: bytes,
        semantic_mask_bytes: bytes,
        reset: bool,
        abstention_threshold: float | None,
    ) -> ServicePrediction:
        rgb = decode_image(rgb_bytes, self.settings.max_upload_bytes, "RGB image")
        semantic_mask = decode_image(
            semantic_mask_bytes, self.settings.max_upload_bytes, "semantic mask"
        )
        started = time.perf_counter()
        raw = self.engine.predict(rgb, semantic_mask)
        temporal, frame_index = self.store.update(sequence_id, raw, reset=reset)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        predicted_id = int(np.argmax(temporal))
        confidence = float(temporal[predicted_id])
        entropy = float(-np.sum(temporal * np.log(np.clip(temporal, 1e-12, 1.0))))
        return ServicePrediction(
            sequence_id=sequence_id,
            frame_index=frame_index,
            predicted_class=self.settings.classes[predicted_id],
            confidence=confidence,
            entropy=entropy,
            abstained=abstention_threshold is not None and confidence < abstention_threshold,
            raw_probabilities=raw,
            temporal_probabilities=temporal,
            inference_ms=elapsed_ms,
        )

    def reset(self, sequence_id: str) -> bool:
        return self.store.reset(sequence_id)


def build_service(settings: Settings) -> WeatherService:
    settings.validate()
    engine: Engine
    if settings.model_mode == "mock":
        engine = MockEngine()
    else:
        engine = DINOv2SkyEngine(settings)
    return WeatherService(settings, engine)
