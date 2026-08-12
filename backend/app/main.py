from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Path, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from .config import Settings
from .predictor import InvalidImageError, WeatherService, build_service
from .schemas import (
    HealthResponse,
    ModelInfo,
    PredictionResponse,
    Probability,
    ResetResponse,
    SegmentationModelInfo,
    SemanticClassArea,
    SemanticSegmentationResponse,
)
from .segmentation import (
    SemanticSegmentationService,
    build_segmentation_service,
    checkpoint_directory,
)
from .services.logging_service import JSONLInferenceLogger

LOGGER = logging.getLogger(__name__)
REQUESTS = Counter("weather_api_predictions_total", "Prediction requests", ["status"])
LATENCY = Histogram(
    "weather_api_inference_seconds",
    "End-to-end prediction latency",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
SEGMENTATION_REQUESTS = Counter(
    "road_api_semantic_segmentation_total",
    "Semantic-segmentation requests",
    ["status"],
)
SEGMENTATION_LATENCY = Histogram(
    "road_api_semantic_segmentation_seconds",
    "End-to-end semantic-segmentation latency",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
WEATHER_ABSTENTIONS = Counter(
    "weather_api_abstentions_total",
    "Weather predictions rejected by the abstention rule",
)
WEATHER_CLASSES = Counter(
    "weather_api_predicted_classes_total",
    "Weather predictions by output class",
    ["predicted_class"],
)
WEATHER_CONFIDENCE = Histogram(
    "weather_api_confidence",
    "EMA confidence of weather predictions",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 1.0),
)
WEATHER_ENTROPY = Histogram(
    "weather_api_entropy",
    "Entropy of EMA weather probabilities",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5),
)
ACTIVE_SEQUENCES = Gauge(
    "weather_api_active_sequences",
    "Number of temporal sequence states currently kept in memory",
)
MODEL_INITIALIZATION_ERRORS = Counter(
    "road_api_model_initialization_errors_total",
    "Model initialization failures",
    ["model"],
)


def create_app(
    settings: Settings | None = None,
    service: WeatherService | None = None,
    segmentation_service: SemanticSegmentationService | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.inference_logger = (
            JSONLInferenceLogger(settings.log_path) if settings.logging_enabled else None
        )
        app.state.startup_error = None
        app.state.segmentation_startup_error = None
        if service is not None:
            app.state.weather_service = service
        else:
            try:
                app.state.weather_service = build_service(settings)
            except Exception as exc:  # readiness exposes failure without hiding liveness
                MODEL_INITIALIZATION_ERRORS.labels(model="weather").inc()
                LOGGER.exception("Phase VI model initialization failed")
                app.state.weather_service = None
                app.state.startup_error = str(exc)
        if segmentation_service is not None:
            app.state.segmentation_service = segmentation_service
        elif settings.segmentation_enabled:
            try:
                app.state.segmentation_service = build_segmentation_service(settings)
            except Exception as exc:
                MODEL_INITIALIZATION_ERRORS.labels(model="segmentation").inc()
                LOGGER.exception("Mask2Former initialization failed")
                app.state.segmentation_service = None
                app.state.segmentation_startup_error = str(exc)
        else:
            app.state.segmentation_service = None
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Research demonstrator combining the Phase VI DINOv2 + EMA weather "
            "pipeline with the fine-tuned Phase VII Mask2Former semantic model."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    def get_service(request: Request) -> WeatherService:
        value = getattr(request.app.state, "weather_service", None)
        if value is None:
            detail = getattr(request.app.state, "startup_error", "model unavailable")
            raise HTTPException(status_code=503, detail=detail)
        return value

    def get_segmentation_service(request: Request) -> SemanticSegmentationService:
        value = getattr(request.app.state, "segmentation_service", None)
        if value is None:
            detail = getattr(
                request.app.state,
                "segmentation_startup_error",
                "semantic-segmentation model unavailable",
            )
            raise HTTPException(status_code=503, detail=detail)
        return value

    @app.get("/health/live", response_model=HealthResponse, tags=["health"])
    def liveness() -> HealthResponse:
        return HealthResponse(status="alive")

    @app.get("/health/ready", response_model=HealthResponse, tags=["health"])
    def readiness(request: Request) -> HealthResponse:
        ready = getattr(request.app.state, "weather_service", None) is not None
        if not ready:
            raise HTTPException(status_code=503, detail="Phase VI artifacts are not ready.")
        segmentation_ready = getattr(request.app.state, "segmentation_service", None) is not None
        return HealthResponse(
            status="ready",
            model_ready=True,
            segmentation_ready=segmentation_ready,
        )

    @app.get("/api/v1/model", response_model=ModelInfo, tags=["model"])
    def model_info(request: Request) -> ModelInfo:
        active = get_service(request)
        return ModelInfo(
            status="ready",
            model_mode=settings.model_mode,
            model=active.engine.name,
            backbone=settings.dino_backbone,
            region=active.engine.region,
            classes=list(settings.classes),
            temporal_method="causal_ema",
            ema_alpha=settings.ema_alpha,
            oracle_mask_required=True,
            limitation=(
                "Validated on Virtual KITTI 2 Scene20 with an oracle semantic-sky mask; "
                "not validated as a deployable real-road safety system."
            ),
        )

    @app.get(
        "/api/v1/segmentation/model",
        response_model=SegmentationModelInfo,
        tags=["segmentation"],
    )
    def segmentation_model_info(request: Request) -> SegmentationModelInfo:
        active = get_segmentation_service(request)
        return SegmentationModelInfo(
            status="ready",
            model=active.engine.name,
            device=active.engine.device_name,
            classes=list(active.engine.labels),
            input_size=active.engine.input_size,
            checkpoint_directory=checkpoint_directory(settings),
            limitation=(
                "Fine-tuned on the KITTI semantic subset. The output is a research "
                "scene interpretation and not a validated road-safety decision."
            ),
        )

    @app.post(
        "/api/v1/segment/semantic",
        response_model=SemanticSegmentationResponse,
        tags=["segmentation"],
    )
    async def segment_semantic(
        request: Request,
        rgb: Annotated[UploadFile, File(description="RGB road-scene image")],
    ) -> SemanticSegmentationResponse:
        active = get_segmentation_service(request)
        try:
            with SEGMENTATION_LATENCY.time():
                result = active.predict(await rgb.read())
        except InvalidImageError as exc:
            SEGMENTATION_REQUESTS.labels(status="invalid").inc()
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            SEGMENTATION_REQUESTS.labels(status="error").inc()
            LOGGER.exception("Semantic segmentation failed")
            raise HTTPException(status_code=500, detail="Semantic segmentation failed.") from exc
        SEGMENTATION_REQUESTS.labels(status="ok").inc()
        return SemanticSegmentationResponse(
            width=result.width,
            height=result.height,
            inference_ms=result.inference_ms,
            model=result.model,
            device=result.device,
            classes_present=[
                SemanticClassArea(
                    class_id=row.class_id,
                    label=row.label,
                    pixels=row.pixels,
                    fraction=row.fraction,
                )
                for row in result.classes_present
            ],
            mask_png_base64=result.mask_png_base64,
            overlay_png_base64=result.overlay_png_base64,
        )

    @app.post("/api/v1/predict/frame", response_model=PredictionResponse, tags=["prediction"])
    async def predict_frame(
        request: Request,
        rgb: Annotated[UploadFile, File(description="RGB road-scene frame")],
        semantic_mask: Annotated[UploadFile, File(description="Matching VKITTI2 colour mask")],
        sequence_id: Annotated[str, Form(min_length=1, max_length=128)],
        reset_sequence: Annotated[bool, Form()] = False,
        abstention_threshold: Annotated[float | None, Form(ge=0.0, le=1.0)] = None,
    ) -> PredictionResponse:
        active = get_service(request)
        threshold = (
            settings.default_abstention_threshold
            if abstention_threshold is None
            else abstention_threshold
        )
        request_id = str(uuid.uuid4())
        try:
            with LATENCY.time():
                result = active.predict(
                    sequence_id=sequence_id,
                    rgb_bytes=await rgb.read(),
                    semantic_mask_bytes=await semantic_mask.read(),
                    reset=reset_sequence,
                    abstention_threshold=threshold,
                )
        except InvalidImageError as exc:
            REQUESTS.labels(status="invalid").inc()
            if request.app.state.inference_logger is not None:
                request.app.state.inference_logger.log(
                    {
                        "request_id": request_id,
                        "event": "invalid_weather_input",
                        "sequence_id": sequence_id,
                        "error": str(exc),
                    }
                )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            REQUESTS.labels(status="error").inc()
            if request.app.state.inference_logger is not None:
                request.app.state.inference_logger.log(
                    {
                        "request_id": request_id,
                        "event": "weather_inference_error",
                        "sequence_id": sequence_id,
                        "error_type": type(exc).__name__,
                    }
                )
            LOGGER.exception("Prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed.") from exc
        REQUESTS.labels(status="ok").inc()
        WEATHER_CLASSES.labels(predicted_class=result.predicted_class).inc()
        WEATHER_CONFIDENCE.observe(result.confidence)
        WEATHER_ENTROPY.observe(result.entropy)
        if result.abstained:
            WEATHER_ABSTENTIONS.inc()
        ACTIVE_SEQUENCES.set(active.store.active_sequence_count())
        if request.app.state.inference_logger is not None:
            request.app.state.inference_logger.log(
                {
                    "request_id": request_id,
                    "event": "weather_inference",
                    "sequence_id": result.sequence_id,
                    "frame_index": result.frame_index,
                    "model": active.engine.name,
                    "region": active.engine.region,
                    "prediction": result.predicted_class,
                    "confidence": result.confidence,
                    "entropy": result.entropy,
                    "abstained": result.abstained,
                    "abstention_threshold": threshold,
                    "ema_alpha": settings.ema_alpha,
                    "inference_ms": result.inference_ms,
                }
            )

        def probability_rows(values):
            return [
                Probability(label=label, value=float(value))
                for label, value in zip(settings.classes, values, strict=True)
            ]

        return PredictionResponse(
            sequence_id=result.sequence_id,
            frame_index=result.frame_index,
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            entropy=result.entropy,
            abstained=result.abstained,
            raw_probabilities=probability_rows(result.raw_probabilities),
            temporal_probabilities=probability_rows(result.temporal_probabilities),
            inference_ms=result.inference_ms,
            model=active.engine.name,
            region=active.engine.region,
            temporal_method=f"EMA(alpha={settings.ema_alpha})",
            request_id=request_id,
            ema_alpha=settings.ema_alpha,
            abstention_threshold=threshold,
        )

    @app.delete(
        "/api/v1/sequences/{sequence_id}", response_model=ResetResponse, tags=["prediction"]
    )
    def reset_sequence(
        request: Request,
        sequence_id: Annotated[str, Path(min_length=1, max_length=128)],
    ) -> ResetResponse:
        active = get_service(request)
        was_present = active.reset(sequence_id)
        ACTIVE_SEQUENCES.set(active.store.active_sequence_count())
        if request.app.state.inference_logger is not None:
            request.app.state.inference_logger.log(
                {
                    "event": "weather_sequence_reset",
                    "sequence_id": sequence_id,
                    "state_existed": was_present,
                }
            )
        return ResetResponse(sequence_id=sequence_id, reset=was_present)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
