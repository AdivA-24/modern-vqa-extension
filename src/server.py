"""
Minimal model-serving surface around the inference engine.

Exposes the three things a fleet operator actually needs from a model node:

- ``GET /healthz``   node state (loading / ready / degraded) plus last-error
- ``GET /metrics``   Prometheus exposition: request counters, latency
                     histograms, tokens/sec, and per-GPU telemetry
                     (utilization, memory, temperature, power) via NVML
- ``POST /v1/vqa``   the actual inference endpoint

GPU telemetry uses pynvml when an NVIDIA GPU is present. Set
``VQA_MOCK_GPU=1`` to emit synthetic NVML-shaped series instead, so the
metrics pipeline can be developed and scraped on machines without a GPU.

Run:  uvicorn src.server:app --port 8000
"""

import os
import random
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

MODEL_NAME = os.environ.get("VQA_MODEL", "llava-hf/llava-v1.6-mistral-7b-hf")
MOCK_GPU = os.environ.get("VQA_MOCK_GPU", "0") == "1"
LAZY_LOAD = os.environ.get("VQA_LAZY_LOAD", "1") == "1"
# Explicit device override (cuda / mps / cpu); default auto-resolves.
DEVICE = os.environ.get("VQA_DEVICE") or None

REQUESTS = Counter("vqa_requests_total", "VQA requests", ["outcome"])
LATENCY = Histogram("vqa_request_seconds", "End-to-end request latency")
TTFT = Histogram("vqa_time_to_first_token_seconds", "Prefill latency")
DECODE_TPS = Gauge("vqa_decode_tokens_per_second", "Decode throughput of last request")
GPU_UTIL = Gauge("gpu_utilization_percent", "GPU utilization", ["gpu"])
GPU_MEM_USED = Gauge("gpu_memory_used_bytes", "GPU memory used", ["gpu"])
GPU_TEMP = Gauge("gpu_temperature_celsius", "GPU temperature", ["gpu"])
GPU_POWER = Gauge("gpu_power_watts", "GPU power draw", ["gpu"])
NODE_STATE = Gauge("vqa_node_state", "Node state (1 = current)", ["state"])


class NodeState:
    """Explicit serving states so orchestration never has to guess."""

    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"

    def __init__(self):
        self._state = self.LOADING
        self._lock = threading.Lock()
        self.last_error = None
        self._publish()

    def _publish(self):
        for s in (self.LOADING, self.READY, self.DEGRADED):
            NODE_STATE.labels(state=s).set(1 if s == self._state else 0)

    def set(self, state, error=None):
        with self._lock:
            self._state = state
            self.last_error = error
            self._publish()

    @property
    def value(self):
        return self._state


state = NodeState()
_engine = None
_engine_lock = threading.Lock()


def get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from src.inference import LlavaInferenceEngine

            state.set(NodeState.LOADING)
            try:
                _engine = LlavaInferenceEngine(model_name=MODEL_NAME, device=DEVICE)
                state.set(NodeState.READY)
            except Exception as exc:
                state.set(NodeState.DEGRADED, error=str(exc))
                raise
        return _engine


def sample_gpu_telemetry():
    """Refresh per-GPU gauges from NVML, or synthesize them under mock mode."""
    if MOCK_GPU:
        GPU_UTIL.labels(gpu="0").set(random.uniform(20, 95))
        GPU_MEM_USED.labels(gpu="0").set(random.uniform(10, 15) * 1024**3)
        GPU_TEMP.labels(gpu="0").set(random.uniform(45, 80))
        GPU_POWER.labels(gpu="0").set(random.uniform(120, 350))
        return
    try:
        import pynvml

        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu = str(i)
            GPU_UTIL.labels(gpu=gpu).set(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            GPU_MEM_USED.labels(gpu=gpu).set(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            GPU_TEMP.labels(gpu=gpu).set(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
            GPU_POWER.labels(gpu=gpu).set(
                pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            )
        pynvml.nvmlShutdown()
    except Exception:
        pass  # no NVML on this host; GPU series simply absent


@asynccontextmanager
async def lifespan(_app):
    if not LAZY_LOAD:
        threading.Thread(target=get_engine, daemon=True).start()
    yield


app = FastAPI(title="modern-vqa-extension serving", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    body = {"state": state.value, "model": MODEL_NAME, "last_error": state.last_error}
    if state.value == NodeState.DEGRADED:
        raise HTTPException(status_code=503, detail=body)
    return body


@app.get("/metrics")
def metrics():
    sample_gpu_telemetry()
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/vqa")
async def vqa(image: UploadFile = File(...), question: str = Form(...)):
    import io

    from PIL import Image

    try:
        engine = get_engine()
    except Exception as exc:
        REQUESTS.labels(outcome="error").inc()
        raise HTTPException(status_code=503, detail=f"model unavailable: {exc}")

    start = time.perf_counter()
    try:
        pil = Image.open(io.BytesIO(await image.read())).convert("RGB")
        result = engine.greedy_decode(pil, question)
    except Exception as exc:
        REQUESTS.labels(outcome="error").inc()
        state.set(NodeState.DEGRADED, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    LATENCY.observe(time.perf_counter() - start)
    TTFT.observe(result.metrics.prefill_seconds)
    DECODE_TPS.set(result.metrics.decode_tokens_per_second)
    REQUESTS.labels(outcome="success").inc()
    return {"answer": result.text, "metrics": result.metrics.as_dict()}
