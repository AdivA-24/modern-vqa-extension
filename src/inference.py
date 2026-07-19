"""
Custom PyTorch inference engine for LLaVA-1.6.

This module implements the token-generation path explicitly instead of
delegating everything to ``model.generate()``:

- explicit device and dtype resolution (CUDA fp16, Apple MPS fp16, CPU fp32)
- a manual greedy decode loop: one prefill forward pass (where LLaVA merges
  image patch embeddings into the prompt), then token-by-token decode steps
  reusing the KV cache via ``past_key_values``
- left-padded batched generation for multi-request throughput
- per-request metrics: time-to-first-token, decode tokens/sec, peak VRAM

Why write this by hand when ``generate()`` exists? Owning the decode loop is
what makes the serving tradeoffs concrete: the prefill/decode split is the
same structure that vLLM's continuous batching and paged KV cache optimize.
See "Relation to vLLM / SGLang" in the README.

The manual loop is verified against ``model.generate()`` output parity in
``tests/test_inference.py`` using a tiny reference checkpoint.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from PIL import Image


def resolve_device_and_dtype(
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[str, torch.dtype]:
    """Pick the best available device and a matching dtype.

    fp16 halves the memory footprint of the 7B weights (~14GB vs ~28GB),
    which is what makes single-GPU inference feasible. CPU stays fp32
    because fp16 CPU kernels are slow or unsupported for many ops.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if dtype is None:
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    return device, dtype


@dataclass
class InferenceMetrics:
    """Timing and memory profile of a single request."""

    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    peak_memory_bytes: Optional[int] = None

    @property
    def time_to_first_token(self) -> float:
        return self.prefill_seconds

    @property
    def decode_tokens_per_second(self) -> float:
        if self.decode_seconds <= 0 or self.generated_tokens <= 1:
            return 0.0
        # First token is produced by prefill; the rest are decode steps.
        return (self.generated_tokens - 1) / self.decode_seconds

    def as_dict(self) -> dict:
        return {
            "prefill_seconds": round(self.prefill_seconds, 4),
            "decode_seconds": round(self.decode_seconds, 4),
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "decode_tokens_per_second": round(self.decode_tokens_per_second, 2),
            "peak_memory_bytes": self.peak_memory_bytes,
        }


@dataclass
class InferenceResult:
    text: str
    metrics: InferenceMetrics = field(default_factory=InferenceMetrics)


class LlavaInferenceEngine:
    """Inference engine for LLaVA-1.6 with an explicit decode loop.

    Either pass a HuggingFace model id, or inject an already-loaded
    ``model`` and ``processor`` (used by the tests with a tiny checkpoint).
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = False,
        model=None,
        processor=None,
    ):
        self.device, self.dtype = resolve_device_and_dtype(device, dtype)
        self.model_name = model_name

        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
        else:
            from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

            self.processor = LlavaNextProcessor.from_pretrained(model_name)
            load_kwargs = dict(
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == "cuda" else None,
            )
            if load_in_4bit:
                # NF4 weight quantization drops the 7B footprint from ~14GB
                # (fp16) to ~5GB, which is what makes a 16GB T4 viable next
                # to the vision tower, activations, and KV cache. Compute
                # still happens in fp16: weights are dequantized per layer.
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name, **load_kwargs
            )
            if self.device != "cuda" and not load_in_4bit:
                self.model = self.model.to(self.device)

        self.model.eval()
        # Decoder-only models must be left-padded for batched generation,
        # otherwise pad tokens sit between the prompt and the new tokens.
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def _build_prompt(self, question: str) -> str:
        return f"USER: <image>\n{question}\nASSISTANT:"

    def _reset_peak_memory(self) -> None:
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def _read_peak_memory(self) -> Optional[int]:
        if self.device == "cuda":
            return int(torch.cuda.max_memory_allocated())
        return None

    @staticmethod
    def _cache_length(past_key_values) -> int:
        """Sequence length currently held in the KV cache.

        Handles both the Cache API (transformers >= 4.36) and the legacy
        tuple-of-tuples layout.
        """
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
        return past_key_values[0][0].shape[2]

    @torch.inference_mode()
    def greedy_decode(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 100,
    ) -> InferenceResult:
        """Generate an answer with a hand-written prefill + decode loop.

        Prefill: one forward pass over the full prompt. For LLaVA this is
        also where image patch embeddings replace the ``<image>`` token, so
        the KV cache that comes out already covers the visual context.

        Decode: each step feeds only the single most recent token plus the
        cache, so per-step cost stays flat instead of re-attending over the
        whole prompt from scratch.
        """
        prompt = self._build_prompt(question)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]

        metrics = InferenceMetrics(prompt_tokens=prompt_tokens)
        self._reset_peak_memory()
        eos_id = self.model.generation_config.eos_token_id
        if isinstance(eos_id, (list, tuple)):
            eos_ids = set(int(t) for t in eos_id)
        else:
            eos_ids = {int(eos_id)} if eos_id is not None else set()

        # --- Prefill ---
        start = time.perf_counter()
        outputs = self.model(**inputs, use_cache=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past = outputs.past_key_values
        metrics.prefill_seconds = time.perf_counter() - start

        generated = [int(next_token[0, 0])]

        # --- Decode loop ---
        start = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if generated[-1] in eos_ids:
                break
            attention_mask = torch.ones(
                (1, self._cache_length(past) + 1), dtype=torch.long, device=self.device
            )
            outputs = self.model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past = outputs.past_key_values
            generated.append(int(next_token[0, 0]))
        metrics.decode_seconds = time.perf_counter() - start

        if generated and generated[-1] in eos_ids:
            generated = generated[:-1]
        metrics.generated_tokens = len(generated)
        metrics.peak_memory_bytes = self._read_peak_memory()

        text = self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return InferenceResult(text=text, metrics=metrics)

    @torch.inference_mode()
    def generate_batch(
        self,
        images: List[Image.Image],
        questions: List[str],
        max_new_tokens: int = 100,
    ) -> List[InferenceResult]:
        """Batched generation: N requests through one left-padded forward pass.

        Static batching amortizes the weight reads across requests, which is
        the single biggest throughput lever on a GPU. Its limit, and the
        reason serving engines exist, is that the whole batch waits for its
        longest member; vLLM's continuous batching removes exactly that.
        """
        if len(images) != len(questions):
            raise ValueError("images and questions must have the same length")

        prompts = [self._build_prompt(q) for q in questions]
        inputs = self.processor(
            text=prompts, images=images, padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]

        self._reset_peak_memory()
        start = time.perf_counter()
        output_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        elapsed = time.perf_counter() - start
        peak = self._read_peak_memory()

        results = []
        new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for row, text in zip(new_tokens, texts):
            metrics = InferenceMetrics(
                prompt_tokens=prompt_tokens,
                generated_tokens=int(row.shape[0]),
                decode_seconds=elapsed,
                peak_memory_bytes=peak,
            )
            results.append(InferenceResult(text=text.strip(), metrics=metrics))
        return results
