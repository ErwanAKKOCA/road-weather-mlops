from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .config import Settings
from .predictor import decode_image

CITYSCAPES_PALETTE = np.asarray(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)


class SemanticEngine(Protocol):
    name: str
    device_name: str
    labels: tuple[str, ...]
    input_size: dict[str, int]

    def segment(self, image: Image.Image) -> np.ndarray: ...


@dataclass(frozen=True)
class SemanticClassResult:
    class_id: int
    label: str
    pixels: int
    fraction: float


@dataclass(frozen=True)
class SemanticSegmentationResult:
    width: int
    height: int
    inference_ms: float
    model: str
    device: str
    classes_present: tuple[SemanticClassResult, ...]
    mask_png_base64: str
    overlay_png_base64: str


def _png_base64(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class Mask2FormerEngine:
    name = "mask2former_kitti_finetuned"

    def __init__(self, settings: Settings):
        model_dir = settings.segmentation_model_dir
        required = {"config.json", "preprocessor_config.json", "model.safetensors"}
        missing = sorted(name for name in required if not (model_dir / name).is_file())
        if missing:
            raise FileNotFoundError(
                f"Missing Mask2Former checkpoint files in {model_dir}: {', '.join(missing)}"
            )

        self._verify_release_manifest(model_dir, settings.segmentation_manifest_filename)

        import torch
        from transformers import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerImageProcessor,
        )

        self.torch = torch
        self.device = self._resolve_device(torch, settings.segmentation_device)
        self.device_name = str(self.device)
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = (
            Mask2FormerForUniversalSegmentation.from_pretrained(model_dir, local_files_only=True)
            .eval()
            .to(self.device)
        )
        id2label = self.model.config.id2label
        self.labels = tuple(str(id2label[index]) for index in range(len(id2label)))
        size = self.processor.size
        self.input_size = {
            "height": int(size.get("height", 384)),
            "width": int(size.get("width", 384)),
        }

    @staticmethod
    def _verify_release_manifest(model_dir: Path, manifest_filename: str) -> None:
        manifest_path = model_dir / manifest_filename
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing Mask2Former release manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = {entry["filename"]: entry for entry in manifest.get("files", [])}
        for filename in ("config.json", "preprocessor_config.json", "model.safetensors"):
            path = model_dir / filename
            expected = entries.get(filename, {}).get("sha256")
            if not expected:
                raise RuntimeError(f"Missing SHA-256 for {filename} in {manifest_path.name}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"SHA-256 mismatch for Mask2Former file {filename}")

    @staticmethod
    def _resolve_device(torch, requested: str):
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SEGMENTATION_DEVICE=cuda but CUDA is unavailable.")
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(requested)

    def segment(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        semantic = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]
        return semantic.detach().cpu().numpy().astype(np.int32, copy=False)


class SemanticSegmentationService:
    def __init__(self, settings: Settings, engine: SemanticEngine):
        self.settings = settings
        self.engine = engine

    def predict(self, image_bytes: bytes) -> SemanticSegmentationResult:
        image = decode_image(image_bytes, self.settings.max_upload_bytes, "RGB image")
        started = time.perf_counter()
        class_map = self.engine.segment(image)
        inference_ms = (time.perf_counter() - started) * 1000.0

        expected_shape = (image.height, image.width)
        if class_map.shape != expected_shape:
            raise RuntimeError(
                f"Segmentation shape {class_map.shape} does not match image {expected_shape}."
            )
        if class_map.size and (
            int(class_map.min()) < 0 or int(class_map.max()) >= len(self.engine.labels)
        ):
            raise RuntimeError("Segmentation output contains an unknown class identifier.")

        colour_mask = CITYSCAPES_PALETTE[class_map]
        rgb = np.asarray(image, dtype=np.uint8)
        alpha = self.settings.segmentation_overlay_alpha
        overlay = np.rint((1.0 - alpha) * rgb + alpha * colour_mask).astype(np.uint8)

        ids, counts = np.unique(class_map, return_counts=True)
        total = int(class_map.size)
        areas = tuple(
            SemanticClassResult(
                class_id=int(class_id),
                label=self.engine.labels[int(class_id)],
                pixels=int(count),
                fraction=float(count / total),
            )
            for class_id, count in zip(ids, counts, strict=True)
        )
        return SemanticSegmentationResult(
            width=image.width,
            height=image.height,
            inference_ms=inference_ms,
            model=self.engine.name,
            device=self.engine.device_name,
            classes_present=areas,
            mask_png_base64=_png_base64(colour_mask),
            overlay_png_base64=_png_base64(overlay),
        )


def build_segmentation_service(settings: Settings) -> SemanticSegmentationService:
    settings.validate()
    return SemanticSegmentationService(settings, Mask2FormerEngine(settings))


def checkpoint_directory(settings: Settings) -> str:
    return str(Path(settings.segmentation_model_dir))
