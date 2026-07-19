# Inference engine walkthrough

A line-by-line explanation of `src/inference.py`: what each decision is, and
why it exists. Written so the code has no magic in it.

## The shape of LLM inference

Generating an answer is two different workloads glued together:

1. **Prefill.** One forward pass over the whole prompt. For LLaVA this is also
   where the vision tower runs: the single `<image>` placeholder token is
   replaced by hundreds of image-patch embeddings before the language model
   sees the sequence. Every token attends to every other token, so this step
   is compute-bound: the GPU's cores are saturated for a short burst. Its
   latency is the user-visible **time-to-first-token (TTFT)**.

2. **Decode.** One token at a time. Each step feeds only the newest token
   through the model and reuses cached attention state for everything before
   it. The arithmetic per step is small, but every step must stream the full
   model weights (~14GB at fp16 for 7B) through the GPU's memory bus to
   produce one token. Decode is therefore **memory-bandwidth-bound**: SM
   utilization can look low while the memory subsystem is the bottleneck.

This split is why "GPU utilization %" alone is a poor health or efficiency
signal for inference workloads, and why the engine reports TTFT and decode
tokens/sec as separate numbers.

## The KV cache

Attention needs the key and value projections of every earlier token. Without
a cache, generating token N means recomputing attention inputs for all N-1
earlier tokens, making generation quadratic. `use_cache=True` makes the model
return `past_key_values`: the stacked K and V tensors per layer. Passing them
back in means each decode step only computes projections for one new token.

The cache is also the main VRAM consumer after the weights, and it grows
linearly with sequence length and batch size. That growth is what OOMs
long-context or high-concurrency serving, and it is the problem vLLM's
PagedAttention exists to manage.

## Walking the code

### `resolve_device_and_dtype`

fp16 on CUDA and Apple MPS, fp32 on CPU. fp16 halves the weight footprint
(~14GB vs ~28GB for 7B), which is the difference between fitting a single
GPU or not. CPU stays fp32 because many CPU kernels lack fast fp16 paths.

### `load_in_4bit`

NF4 weight quantization (bitsandbytes) shrinks weights to ~5GB. Compute still
happens in fp16: each layer's weights are dequantized on the fly. This trades
some decode speed (dequantization work) for fitting a 16GB card such as a
Colab T4 with room left for the vision tower, activations, and KV cache.

### `padding_side = "left"` (in `__init__`)

Decoder-only models generate after the last token of the sequence. If a batch
right-padded its shorter prompts, generated tokens would be appended after
pad tokens. Left padding puts all prompts flush against the generation
boundary so one batched forward pass is correct for every row.

### `greedy_decode`: the prefill block

```python
outputs = self.model(**inputs, use_cache=True)
next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
past = outputs.past_key_values
```

One forward pass over prompt plus image. The logits row at position -1 is the
distribution for the next token; `argmax` is greedy decoding (deterministic,
which is also what makes the parity test meaningful). The KV cache that comes
back already covers the expanded image tokens.

### `greedy_decode`: the decode loop

```python
attention_mask = torch.ones((1, self._cache_length(past) + 1), ...)
outputs = self.model(input_ids=next_token, attention_mask=attention_mask,
                     past_key_values=past, use_cache=True)
```

Only the single newest token goes in; the cache supplies everything else.
The attention mask must cover cache length plus one because attention runs
over all cached positions and the new one. `_cache_length` handles both the
modern `Cache` API and the legacy tuple layout. `pixel_values` is never passed
again: the image is already baked into the cache.

The loop stops on EOS (the generation config may define several EOS ids, so
membership is checked against a set) or at `max_new_tokens`.

### `generate_batch`

Static batching through HF `generate()`: N requests in one left-padded call.
Weight reads during decode are amortized across the batch, which is the
single biggest throughput lever on a GPU. Its limitation is that the whole
batch finishes when its longest member does; short requests wait for long
ones. Removing exactly that limitation (admitting and retiring requests at
every decode step) is vLLM's continuous batching.

### `InferenceMetrics`

TTFT and decode tokens/sec are the two service-level numbers of an inference
node; peak memory is captured because VRAM headroom, not compute, is usually
what kills serving workloads. `decode_tokens_per_second` divides by
`generated_tokens - 1` because the first token comes out of prefill, not the
decode loop.

## What the parity test proves

`tests/test_inference.py` runs the manual loop and HF `generate(do_sample=False)`
on the same inputs with a tiny LLaVA-Next checkpoint and asserts identical
output text. Greedy decoding is deterministic, so any error in cache handling,
masking, or token selection would diverge within a few tokens. It proves the
loop is a correct implementation; it says nothing about performance, which is
what the GPU benchmark notebook measures.

## Relation to vLLM and SGLang

This engine is one request (or one static batch) at a time with a naive,
contiguously grown KV cache. Production serving engines change exactly those
two things:

- **vLLM**: continuous batching (requests join and leave the running batch at
  decode-step granularity) and PagedAttention (KV cache stored in fixed-size
  blocks with an indirection table, like virtual memory, eliminating
  fragmentation and enabling much larger effective batch sizes), plus prefix
  caching and tensor parallelism.
- **SGLang**: RadixAttention (a prefix tree over KV cache so requests sharing
  a prompt prefix, such as a common system prompt, reuse cache across
  requests) and fast constrained/structured decoding.

Nobody forks these to serve a standard model; you run them and keep your own
code at the prompt-construction and pre/post-processing layer. Forks exist at
the hardware-backend layer (for example vLLM's TPU backend) and at the
operator-tooling layer (metrics exporters, remediation agents), not at the
"my model" layer.
