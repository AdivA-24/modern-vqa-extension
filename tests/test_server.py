"""
Smoke tests for the serving surface, using the tiny checkpoint and mocked
GPU telemetry so they run anywhere.
"""

import io
import os

os.environ["VQA_MODEL"] = "trl-internal-testing/tiny-LlavaNextForConditionalGeneration"
os.environ["VQA_MOCK_GPU"] = "1"
os.environ["VQA_LAZY_LOAD"] = "1"
os.environ["VQA_DEVICE"] = "cpu"

from fastapi.testclient import TestClient
from PIL import Image

from src.server import app

client = TestClient(app)


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (96, 96), color=(200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_healthz_reports_state():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["state"] in ("loading", "ready")


def test_metrics_exposition():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "vqa_requests_total" in body
    assert "gpu_utilization_percent" in body  # mock mode emits GPU series
    assert "vqa_node_state" in body


def test_vqa_request_end_to_end():
    resp = client.post(
        "/v1/vqa",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        data={"question": "What color is this image?"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "answer" in body
    assert body["metrics"]["generated_tokens"] > 0

    # After a successful request the node must report ready.
    assert client.get("/healthz").json()["state"] == "ready"
