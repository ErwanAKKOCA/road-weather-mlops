from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.predictor import build_service


def make_png(value: int) -> bytes:
    image = Image.new("RGB", (96, 64), (value, value, value))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_mask() -> bytes:
    # MockEngine ignores the semantic content, but the API still requires a valid image.
    return make_png(127)


def make_client() -> TestClient:
    settings = Settings(
        model_mode="mock",
        segmentation_enabled=False,
        default_abstention_threshold=0.90,
        logging_enabled=False,
    )
    return TestClient(create_app(settings=settings, service=build_service(settings)))


def post_frame(client, content, sequence_id="route-01", threshold="0.90"):
    return client.post(
        "/api/v1/predict/frame",
        files={
            "rgb": ("frame.png", content, "image/png"),
            "semantic_mask": ("mask.png", make_mask(), "image/png"),
        },
        data={
            "sequence_id": sequence_id,
            "reset_sequence": "false",
            "abstention_threshold": threshold,
        },
    )


def values(rows):
    return np.asarray([row["value"] for row in rows], dtype=float)


def test_health_model_info_and_prometheus_metrics():
    with make_client() as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        model = client.get("/api/v1/model")
        assert model.status_code == 200
        assert model.json()["model_mode"] == "mock"

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "weather_api_predictions_total" in metrics.text
        assert "weather_api_abstentions_total" in metrics.text


def test_invalid_upload_is_rejected():
    with make_client() as client:
        response = client.post(
            "/api/v1/predict/frame",
            files={
                "rgb": ("bad.txt", b"not-an-image", "text/plain"),
                "semantic_mask": ("mask.png", make_mask(), "image/png"),
            },
            data={"sequence_id": "bad-test", "abstention_threshold": "0.90"},
        )
        assert response.status_code == 422


def test_mock_inference_is_deterministic_after_reset():
    frame = make_png(100)
    with make_client() as client:
        first = post_frame(client, frame, sequence_id="deterministic")
        assert first.status_code == 200
        reset = client.delete("/api/v1/sequences/deterministic")
        assert reset.status_code == 200
        second = post_frame(client, frame, sequence_id="deterministic")
        assert second.status_code == 200
        assert first.json()["raw_probabilities"] == second.json()["raw_probabilities"]


def test_ema_changes_second_frame():
    with make_client() as client:
        first = post_frame(client, make_png(40), sequence_id="ema-test").json()
        second = post_frame(client, make_png(220), sequence_id="ema-test").json()
        assert first["frame_index"] == 0
        assert second["frame_index"] == 1
        assert not np.allclose(
            values(second["raw_probabilities"]), values(second["temporal_probabilities"])
        )


def test_sequences_are_isolated():
    with make_client() as client:
        post_frame(client, make_png(30), sequence_id="seq-A")
        first_b = post_frame(client, make_png(200), sequence_id="seq-B").json()
        assert first_b["frame_index"] == 0
        np.testing.assert_allclose(
            values(first_b["raw_probabilities"]), values(first_b["temporal_probabilities"])
        )


def test_reset_restarts_frame_index_and_ema():
    with make_client() as client:
        post_frame(client, make_png(20), sequence_id="reset-test")
        post_frame(client, make_png(180), sequence_id="reset-test")
        reset = client.delete("/api/v1/sequences/reset-test")
        assert reset.status_code == 200
        assert reset.json()["reset"] is True
        after = post_frame(client, make_png(180), sequence_id="reset-test").json()
        assert after["frame_index"] == 0
        np.testing.assert_allclose(
            values(after["raw_probabilities"]), values(after["temporal_probabilities"])
        )


def test_abstention_threshold_is_applied():
    with make_client() as client:
        result = post_frame(
            client, make_png(123), sequence_id="abstention-test", threshold="1.0"
        ).json()
        assert result["abstained"] is True
        assert result["abstention_threshold"] == 1.0


def test_default_abstention_threshold_is_centralized():
    settings = Settings(
        model_mode="mock",
        segmentation_enabled=False,
        default_abstention_threshold=0.90,
        logging_enabled=False,
    )
    app = create_app(settings=settings, service=build_service(settings))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict/frame",
            files={
                "rgb": ("frame.png", make_png(123), "image/png"),
                "semantic_mask": ("mask.png", make_mask(), "image/png"),
            },
            data={"sequence_id": "default-threshold"},
        )
        assert response.status_code == 200
        assert response.json()["abstention_threshold"] == 0.90
