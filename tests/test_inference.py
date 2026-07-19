"""
Verifies the hand-written decode loop against HuggingFace ``generate()``.

Uses a tiny random-weight LLaVA-Next checkpoint so the test runs on CPU in
seconds. Random weights are fine here: greedy decoding is deterministic, so
if the manual prefill + KV-cache loop is implemented correctly it must
produce token-for-token identical output to ``generate(do_sample=False)``.
"""

import pytest
import torch
from PIL import Image

TINY_RANDOM = "trl-internal-testing/tiny-LlavaNextForConditionalGeneration"


@pytest.fixture(scope="module")
def engine():
    from transformers import AutoProcessor, LlavaNextForConditionalGeneration

    from src.inference import LlavaInferenceEngine

    processor = AutoProcessor.from_pretrained(TINY_RANDOM)
    model = LlavaNextForConditionalGeneration.from_pretrained(TINY_RANDOM)
    return LlavaInferenceEngine(
        model_name=TINY_RANDOM, device="cpu", model=model, processor=processor
    )


@pytest.fixture()
def image():
    return Image.new("RGB", (96, 96), color=(120, 40, 200))


def test_greedy_decode_matches_hf_generate(engine, image):
    question = "What color is this image?"
    max_new_tokens = 12

    result = engine.greedy_decode(image, question, max_new_tokens=max_new_tokens)

    prompt = engine._build_prompt(question)
    inputs = engine.processor(text=prompt, images=image, return_tensors="pt")
    with torch.inference_mode():
        output_ids = engine.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    expected = engine.processor.tokenizer.decode(
        new_tokens, skip_special_tokens=True
    ).strip()

    assert result.text == expected


def test_metrics_populated(engine, image):
    result = engine.greedy_decode(image, "Describe the image.", max_new_tokens=6)
    m = result.metrics
    assert m.prompt_tokens > 0
    assert 0 < m.generated_tokens <= 6
    assert m.prefill_seconds > 0
    assert m.as_dict()["generated_tokens"] == m.generated_tokens


def test_batch_shapes(engine, image):
    questions = ["What is this?", "What color dominates?"]
    results = engine.generate_batch([image, image], questions, max_new_tokens=6)
    assert len(results) == 2
    for r in results:
        assert isinstance(r.text, str)
        assert r.metrics.generated_tokens > 0
