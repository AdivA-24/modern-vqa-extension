# Modern VQA Extension: Cross-Lingual Visual Question Answering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)

Cross-lingual Visual Question Answering: ask a question about an image in any
of 100+ languages, get the answer in the language you choose. This repo
extends an earlier Rice University COMP646 proof-of-concept
([original demo](https://huggingface.co/spaces/ixxan/multilingual-vqa)) by
replacing its ViLT-era answering stage with modern vision-language models
(LLaVA-1.6), and adds a custom PyTorch inference engine with an explicit
prefill/decode loop, a serving surface with Prometheus metrics, and a
real-GPU benchmark notebook.

## Motivation

Most multilingual VQA models are trained on a handful of high-resource
languages and generalize poorly outside them. This project's design goal,
motivated by low-resource languages such as Uyghur and Kurdish (the original
project was co-authored by a native Uyghur speaker), is to make answer
quality independent of the question language: translate in, answer with the
strongest available English-centric model, translate out. The system then
supports any language the translation layer supports, including ones no VQA
training set covers.

```
Question (any language) -> Google Translate -> English
                                                |
Image + English question -> VLM (LLaVA-1.6 / ViLT / BLIP-2) -> Answer
                                                |
Answer -> Google Translate -> Target language
```

## Project timeline

- **Class project (Rice COMP646)**: proof-of-concept cross-lingual VQA with
  ViLT + translation + FLAN-T5 answer composition
  ([original demo](https://huggingface.co/spaces/ixxan/multilingual-vqa)).
- **Oct-Nov 2025 rework** (this repo's early history): rebuilt around modern
  VLMs: LLaVA-1.6 integration, unified pipeline, ViLT/BLIP-2 baselines,
  Gradio comparison demo, evaluation rubric.
- **July 2026 hardening**: custom inference engine with an explicit
  prefill/decode loop and parity tests, serving surface with Prometheus
  metrics, real-GPU benchmark notebook.

## What's in the repo

| Layer | File | What it does |
|-------|------|--------------|
| Pipeline | `src/pipeline.py` | Orchestrates translate -> VQA -> translate-back |
| Models | `src/models.py` | Wrappers for LLaVA-1.6 (2024), BLIP-2 (2023), ViLT (2021) |
| Translation | `src/translation.py` | Google Translate integration, 100+ languages |
| **Inference engine** | `src/inference.py` | Hand-written prefill + KV-cache decode loop, batched generation, per-request metrics |
| **Serving** | `src/server.py` | FastAPI: `/healthz` state machine, Prometheus `/metrics` with GPU telemetry, `/v1/vqa` |
| Benchmark | `notebooks/gpu_benchmark.ipynb` | Run-all Colab notebook: real-GPU latency/throughput/telemetry measurements |
| Tests | `tests/` | Decode-loop parity vs HF `generate()`, serving smoke tests |
| Demo | `demo/app.py` | Gradio comparison UI (ViLT vs LLaVA-1.6) |

## The custom inference engine

`src/inference.py` implements the generation path explicitly instead of
delegating to `model.generate()`:

- **Prefill**: one forward pass over prompt + image (this is where LLaVA
  swaps the `<image>` token for image-patch embeddings), producing the first
  token and the KV cache. Compute-bound; its latency is time-to-first-token.
- **Decode**: token-by-token steps that feed only the newest token plus the
  cache. Memory-bandwidth-bound: each step streams the full weights to emit
  one token, which is why raw GPU utilization is a misleading signal for
  inference workloads.
- **Batched generation**: left-padded static batching to amortize weight
  reads across requests.
- **Metrics**: TTFT, decode tokens/sec, and peak VRAM per request.
- **Memory management**: fp16 on GPU (~14GB for 7B weights), optional 4-bit
  NF4 quantization (~5GB) to fit 16GB cards like a Colab T4.

The manual loop is verified **token-for-token identical** to HF
`generate(do_sample=False)` in `tests/test_inference.py`, using a tiny
LLaVA-Next checkpoint so the test runs on CPU in seconds.

Full annotated tour: [docs/INFERENCE-WALKTHROUGH.md](docs/INFERENCE-WALKTHROUGH.md).

### Relation to vLLM / SGLang

This engine is deliberately the naive baseline: one request (or one static
batch) at a time, contiguously grown KV cache. Production serving engines
change exactly those two things: vLLM adds continuous batching (requests
join/leave the running batch at decode-step granularity) and PagedAttention
(block-based KV cache, like virtual memory); SGLang adds RadixAttention
(cross-request KV reuse for shared prefixes) and fast constrained decoding.
At serving scale you run one of those and keep code like this repo's at the
prompt-construction and pre/post-processing layer. The walkthrough doc covers
this in more depth.

## Serving and observability

```bash
uvicorn src.server:app --port 8000
```

- `GET /healthz`: node state (`loading` / `ready` / `degraded`) plus last error
- `GET /metrics`: Prometheus exposition: request counters, latency and TTFT
  histograms, decode tokens/sec, and per-GPU telemetry (utilization, memory,
  temperature, power) via NVML
- `POST /v1/vqa`: multipart `image` + `question` form fields

On machines without an NVIDIA GPU, set `VQA_MOCK_GPU=1` to emit synthetic
GPU series so the metrics pipeline can be developed and scraped anywhere.
Other knobs: `VQA_MODEL` (HF model id), `VQA_DEVICE` (cuda/mps/cpu),
`VQA_LAZY_LOAD=0` to load the model at startup.

Minimal Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: vqa-node
    scrape_interval: 15s
    static_configs:
      - targets: ["localhost:8000"]
```

## Quick start

```bash
git clone https://github.com/AdivA-24/modern-vqa-extension.git
cd modern-vqa-extension
pip install -r requirements.txt
```

```python
from PIL import Image
from src.pipeline import CrossLingualVQAPipeline

pipeline = CrossLingualVQAPipeline("modern")   # LLaVA-1.6
result = pipeline.query(
    image=Image.open("path/to/image.jpg"),
    question="¿Qué están haciendo los gatos?",  # any language
    target_lang="es",
)
print(result["answer"])
```

Direct engine use (no translation layer):

```python
from PIL import Image
from src.inference import LlavaInferenceEngine

engine = LlavaInferenceEngine(load_in_4bit=True)  # fits a 16GB T4
result = engine.greedy_decode(Image.open("cat.jpg"), "What is the cat doing?")
print(result.text)
print(result.metrics.as_dict())  # TTFT, decode tok/s, peak VRAM
```

## Model comparison: 2021 vs 2024

Qualitative comparison of ViLT (`dandelin/vilt-b32-finetuned-vqa`, 113M
params, classification over 3,129 answers) against LLaVA-1.6
(`llava-hf/llava-v1.6-mistral-7b-hf`, 7B params, generative):

| Question | ViLT (2021) | LLaVA-1.6 (2024) |
|----------|-------------|------------------|
| "What are the cats doing?" | "sleeping" | "The two cats are lying together on a couch, appearing to be resting or sleeping. They seem comfortable and relaxed in each other's company." |
| "¿Cuántos gatos hay?" (via translation) | "2" | "Hay dos gatos en la imagen, descansando juntos en el sofá." |

Scored against a rubric (answer completeness, correctness, reasoning) on our
own evaluation prompts, LLaVA-1.6 improves answer quality on the order of
85% over the ViLT baseline. This is a rubric score on an internal evaluation
set, not a public benchmark number; the step change is from classification
over a fixed answer vocabulary to open-ended generation with reasoning.

## GPU benchmark

`notebooks/gpu_benchmark.ipynb` is a run-all Colab notebook (free T4) that
loads the 7B model 4-bit through the engine and records real TTFT, decode
tokens/sec, peak VRAM, sequential-vs-batched throughput, and NVML telemetry
sampled separately over the prefill and decode windows.

<!-- RESULTS:BEGIN -->
Results from a live run land here (see `gpu_run_results.json`).
<!-- RESULTS:END -->

## Tests

```bash
pytest tests/
```

- `test_inference.py`: manual decode loop produces identical output to HF
  `generate()`; metrics populate; batch shapes are correct.
- `test_server.py`: health states, Prometheus exposition, end-to-end request
  against a tiny checkpoint with mocked GPU telemetry.

## Requirements

- Python 3.9+, PyTorch 2.0+, Transformers 4.41+
- GPU recommended for the 7B model: 24GB at fp16, or 16GB with
  `load_in_4bit=True`. CPU inference works but is slow.
- Known limitation: `googletrans==4.0.0rc1` is unofficial and can be flaky;
  the translation layer isolates it behind `src/translation.py` so it can be
  swapped for an official API.

## Citation

Original proof-of-concept this repo extends:

```bibtex
@misc{abdurahman2023crosslingualvqa,
  title={Multilingual Visual Question Answering with Cross-lingual Support},
  author={Abdurahman, Irpan and Ahsan, Adiv},
  year={2023},
  howpublished={Rice University COMP646 project},
  note={Demo: https://huggingface.co/spaces/ixxan/multilingual-vqa}
}
```

## Acknowledgments

LLaVA team, Hugging Face, the ViLT authors, and Rice University COMP646.

## Contact

Adiv Ahsan - adiv.ahsan1@gmail.com
