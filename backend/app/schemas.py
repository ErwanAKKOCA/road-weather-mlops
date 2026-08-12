from __future__ import annotations

from pydantic import BaseModel, Field


class Probability(BaseModel):
    label: str
    value: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------
# Weather prediction
# ---------------------------------------------------------------------


class PredictionResponse(BaseModel):
    sequence_id: str
    frame_index: int = Field(ge=0)

    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    entropy: float = Field(ge=0.0)
    abstained: bool

    raw_probabilities: list[Probability]
    temporal_probabilities: list[Probability]

    inference_ms: float = Field(ge=0.0)

    model: str
    region: str
    temporal_method: str

    oracle_mask_required: bool = True

    # MLOps / traceability.
    # Optional so existing code remains compatible.
    request_id: str | None = None
    ema_alpha: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
    )
    abstention_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


# ---------------------------------------------------------------------
# Weather model metadata
# ---------------------------------------------------------------------


class ModelInfo(BaseModel):
    status: str
    model_mode: str
    model: str
    backbone: str
    region: str
    classes: list[str]

    temporal_method: str
    ema_alpha: float

    oracle_mask_required: bool
    limitation: str


# ---------------------------------------------------------------------
# Operational endpoints
# ---------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    model_ready: bool | None = None
    segmentation_ready: bool | None = None


class ResetResponse(BaseModel):
    sequence_id: str
    reset: bool


# ---------------------------------------------------------------------
# Semantic segmentation
# ---------------------------------------------------------------------


class SemanticClassArea(BaseModel):
    class_id: int = Field(ge=0)
    label: str

    pixels: int = Field(ge=0)
    fraction: float = Field(
        ge=0.0,
        le=1.0,
    )


class SemanticSegmentationResponse(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    inference_ms: float = Field(ge=0.0)

    model: str
    device: str

    classes_present: list[SemanticClassArea]

    mask_png_base64: str
    overlay_png_base64: str


class SegmentationModelInfo(BaseModel):
    status: str
    model: str
    device: str

    classes: list[str]
    input_size: dict[str, int]

    checkpoint_directory: str
    limitation: str
