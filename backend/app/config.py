from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class Settings:
    # ---------------------------------------------------------
    # Application
    # ---------------------------------------------------------
    app_name: str = "Phase VIII Integrated Road-Scene API"
    app_version: str = "0.2.0"

    # ---------------------------------------------------------
    # Weather model
    # ---------------------------------------------------------
    model_mode: str = os.getenv(
        "MODEL_MODE",
        "phase6",
    )

    artifact_dir: Path = Path(
        os.getenv(
            "ARTIFACT_DIR",
            "/artifacts",
        )
    )

    model_filename: str = os.getenv(
        "MODEL_FILENAME",
        "dinov2_logistic_semantic_sky_v3_1.joblib",
    )

    region_filename: str = os.getenv(
        "REGION_FILENAME",
        "region_definitions_v3_1.json",
    )

    release_manifest_filename: str = os.getenv(
        "RELEASE_MANIFEST_FILENAME",
        "release_manifest_v4_1.json",
    )

    dino_backbone: str = os.getenv(
        "DINO_BACKBONE",
        "vit_small_patch14_dinov2.lvd142m",
    )

    classes: tuple[str, ...] = _csv_env(
        "CLASS_ORDER",
        "clone,fog,rain",
    )

    # ---------------------------------------------------------
    # Temporal inference
    # ---------------------------------------------------------
    ema_alpha: float = float(
        os.getenv(
            "EMA_ALPHA",
            "0.6",
        )
    )

    sequence_ttl_seconds: int = int(
        os.getenv(
            "SEQUENCE_TTL_SECONDS",
            "3600",
        )
    )

    max_sequences: int = int(
        os.getenv(
            "MAX_SEQUENCES",
            "512",
        )
    )

    # ---------------------------------------------------------
    # Selective prediction / abstention
    # ---------------------------------------------------------
    default_abstention_threshold: float = float(
        os.getenv(
            "DEFAULT_ABSTENTION_THRESHOLD",
            "0.90",
        )
    )

    # ---------------------------------------------------------
    # Image processing
    # ---------------------------------------------------------
    neutral_background: int = int(
        os.getenv(
            "NEUTRAL_BACKGROUND",
            "127",
        )
    )

    max_upload_bytes: int = int(
        os.getenv(
            "MAX_UPLOAD_BYTES",
            str(12 * 1024 * 1024),
        )
    )

    # ---------------------------------------------------------
    # Semantic segmentation
    # ---------------------------------------------------------
    segmentation_enabled: bool = _bool_env(
        "SEGMENTATION_ENABLED",
        False,
    )

    segmentation_model_dir: Path = Path(
        os.getenv(
            "SEGMENTATION_MODEL_DIR",
            "/artifacts/mask2former_kitti_best",
        )
    )

    segmentation_device: str = os.getenv(
        "SEGMENTATION_DEVICE",
        "auto",
    )

    segmentation_manifest_filename: str = os.getenv(
        "SEGMENTATION_MANIFEST_FILENAME",
        "release_manifest.json",
    )

    segmentation_overlay_alpha: float = float(
        os.getenv(
            "SEGMENTATION_OVERLAY_ALPHA",
            "0.55",
        )
    )

    # ---------------------------------------------------------
    # Logging / observability
    # ---------------------------------------------------------
    logging_enabled: bool = _bool_env(
        "LOGGING_ENABLED",
        True,
    )

    log_path: Path = Path(
        os.getenv(
            "LOG_PATH",
            "/tmp/road-weather-lab/inference.jsonl",
        )
    )

    # ---------------------------------------------------------
    # API / CORS
    # ---------------------------------------------------------
    cors_origins: tuple[str, ...] = _csv_env(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:8080",
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    def validate(self) -> None:
        if self.model_mode not in {
            "phase6",
            "mock",
        }:
            raise ValueError("MODEL_MODE must be 'phase6' or 'mock'.")

        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("EMA_ALPHA must be in (0, 1].")

        if len(self.classes) < 2:
            raise ValueError("At least two classes are required.")

        if not 0.0 <= self.default_abstention_threshold <= 1.0:
            raise ValueError("DEFAULT_ABSTENTION_THRESHOLD must be in [0, 1].")

        if self.sequence_ttl_seconds <= 0:
            raise ValueError("SEQUENCE_TTL_SECONDS must be > 0.")

        if self.max_sequences <= 0:
            raise ValueError("MAX_SEQUENCES must be > 0.")

        if self.max_upload_bytes <= 0:
            raise ValueError("MAX_UPLOAD_BYTES must be > 0.")

        if self.segmentation_device not in {
            "auto",
            "cpu",
            "cuda",
        }:
            raise ValueError("SEGMENTATION_DEVICE must be 'auto', 'cpu', or 'cuda'.")

        if not 0.0 <= self.segmentation_overlay_alpha <= 1.0:
            raise ValueError("SEGMENTATION_OVERLAY_ALPHA must be in [0, 1].")
