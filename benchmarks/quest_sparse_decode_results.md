# Quest Sparse Decode Benchmark Notes

This file records reproducible checkpoints for the experimental AlayaJet Quest
Metal sparse decode path. Commands use:

```bash
.venv-mlx/bin/python benchmarks/quest_sparse_decode_benchmark.py \
  --model mlx-community/Qwen3-4B-4bit-DWQ-053125 \
  --target-prompt-tokens 32768 \
  --page-budget 64 \
  --max-tokens 16
```

## 2026-05-11 Baseline

Environment:

- MLX: `0.30.5.dev20260511+e8001a24`
- Model: `mlx-community/Qwen3-4B-4bit-DWQ-053125`
- Prompt length: 33,619 tokens
- Quest page size: 64
- Quest page budget: 64

Default Quest path:

```json
{
  "case": "default",
  "elapsed_s": 327.9179,
  "ttft_s": 317.8861,
  "tpot_ms": 714.101,
  "prompt_tps": 105.792,
  "generation_tps": 1.5,
  "prompt_tokens": 33619,
  "completion_tokens": 15,
  "page_budget": 64
}
```

Quest timing summary:

```text
decode_append_kv: 0.788s, avg 1.368ms, n=576
decode_estimate_topk: 1.043s, avg 1.811ms, n=576
decode_sparse_attn: 13.541s, avg 23.509ms, n=576
decode_disk_read: 6.758s, avg 11.733ms, n=576
decode_stream_attn: 13.269s, avg 23.036ms, n=576
```

## Experimental Fused Metal Path

The fused path is enabled with:

```bash
ALAYAJET_QUEST_METAL_SPARSE=1
```

The fused path also supports an approximate top-k budget independent of the
Quest controller budget:

```bash
ALAYAJET_QUEST_METAL_PAGE_BUDGET=8
```

It currently fuses page scoring, top-k selection, selected-page attention, and
last-page attention into one custom Metal kernel. The first implementation was
correct but slower than default Quest because it materialized all candidate KV
pages for every layer and decode step.

After adding a per-layer candidate KV cache, the short-context TPOT checkpoint
improved:

```json
{
  "case": "default",
  "prompt_tokens": 1699,
  "page_budget": 4,
  "tpot_ms": 143.277,
  "generation_tps": 9.306
}
```

```json
{
  "case": "fused_metal",
  "prompt_tokens": 1699,
  "page_budget": 4,
  "tpot_ms": 59.253,
  "generation_tps": 22.501
}
```

The 32k fused path is not yet competitive. Before softmax-weight reuse, it
measured:

```json
{
  "case": "fused_metal",
  "prompt_tokens": 33619,
  "page_budget": 64,
  "tpot_ms": 5615.223,
  "generation_tps": 0.191
}
```

A synthetic kernel-only benchmark after softmax-weight reuse:

```text
Hq=32, Hkv=8, head_dim=128, pages=512, page_size=64, page_budget=64
avg fused kernel time: 75.606ms
```

This is still slower than the default path's 32k `decode_sparse_attn` average
of 23.509ms per layer/step, so the current fused kernel remains experimental.

Attempted optimization: split each head into multiple output-dimension block
threadgroups to increase occupancy. Result:

```text
Hq=32, Hkv=8, head_dim=128, pages=512, page_size=64, page_budget=64
avg fused kernel time: 303.450ms
```

This regressed badly because page scoring and token logits were duplicated for
each dimension block. The change was reverted.

## 32k TPOT With Approximate Top-8

Command:

```bash
.venv-mlx/bin/python benchmarks/quest_sparse_decode_benchmark.py \
  --model mlx-community/Qwen3-4B-4bit-DWQ-053125 \
  --target-prompt-tokens 32768 \
  --page-budget 64 \
  --fused-page-budget 8 \
  --max-tokens 16 \
  --cases fused
```

Result:

```json
{
  "case": "fused_metal",
  "prompt_tokens": 33619,
  "page_budget": 64,
  "fused_page_budget": 8,
  "ttft_s": 333.801,
  "tpot_ms": 111.257,
  "generation_tps": 9.629
}
```

Quest timing summary:

```text
decode_append_kv: 0.618s, avg 1.073ms, n=576
decode_sparse_attn: 17.525s, avg 30.425ms, n=576
```

Fair top-8 default Quest comparison:

```bash
.venv-mlx/bin/python benchmarks/quest_sparse_decode_benchmark.py \
  --model mlx-community/Qwen3-4B-4bit-DWQ-053125 \
  --target-prompt-tokens 32768 \
  --page-budget 8 \
  --max-tokens 16 \
  --cases default
```

```json
{
  "case": "default",
  "prompt_tokens": 33619,
  "page_budget": 8,
  "ttft_s": 372.4605,
  "tpot_ms": 343.692,
  "generation_tps": 3.117
}
```

The fused approximate top-8 path is therefore:

- 6.4x faster TPOT than default Quest top-64: `714.101 / 111.257`.
- 3.1x faster TPOT than default Quest top-8: `343.692 / 111.257`.

The tradeoff is recall/quality: this is approximate top-8 while the original
baseline uses top-64. It needs quality/regression evaluation before becoming
the default.

Next optimization targets at this checkpoint:

- Reduce or eliminate all-candidate KV materialization for the fused path.
- Improve the in-kernel page selection loop and token scoring path.
- Keep TPOT as the primary gate; TTFT is expected to rise if decode builds a
  resident KV view, but steady-state TPOT must improve.

The following section includes the page-selection optimization result.

## 32k TPOT Budget Scaling

The 32k benchmark was repeated with a larger fused budget to measure the TPOT
cost of a more conservative sparse decode setting:

```bash
.venv-mlx/bin/python benchmarks/quest_sparse_decode_benchmark.py \
  --model mlx-community/Qwen3-4B-4bit-DWQ-053125 \
  --target-prompt-tokens 32768 \
  --page-budget 64 \
  --fused-page-budget 16 \
  --max-tokens 16 \
  --cases fused
```

Initial top-16 result:

```json
{
  "case": "fused_metal",
  "prompt_tokens": 33619,
  "page_budget": 64,
  "fused_page_budget": 16,
  "ttft_s": 368.4559,
  "tpot_ms": 251.94,
  "generation_tps": 4.252
}
```

After changing the V accumulation stage so each output dimension is reduced by
multiple lanes within the same threadgroup, the top-16 result was:

```json
{
  "case": "fused_metal",
  "prompt_tokens": 33619,
  "page_budget": 64,
  "fused_page_budget": 16,
  "ttft_s": 377.4729,
  "tpot_ms": 246.906,
  "generation_tps": 4.339
}
```

The same kernel change was also checked against the current best TPOT setting,
fused top-8:

```json
{
  "case": "fused_metal",
  "prompt_tokens": 33619,
  "page_budget": 64,
  "fused_page_budget": 8,
  "ttft_s": 398.5785,
  "tpot_ms": 112.11,
  "generation_tps": 9.555
}
```

The next kernel change removed the repeated `selected` scan from in-kernel
top-k selection. Instead, the selected page score is overwritten with
`-INFINITY`, reducing selection from roughly `O(k^2 * pages)` to
`O(k * pages)`.

Fused top-16 after the faster in-kernel top-k:

```json
{
  "case": "fused_metal",
  "prompt_tokens": 33619,
  "page_budget": 64,
  "fused_page_budget": 16,
  "ttft_s": 324.4117,
  "tpot_ms": 86.317,
  "generation_tps": 12.41
}
```

Fused top-8 after the same top-k change:

```json
{
  "case": "fused_metal",
  "prompt_tokens": 33619,
  "page_budget": 64,
  "fused_page_budget": 8,
  "ttft_s": 379.9329,
  "tpot_ms": 75.629,
  "generation_tps": 14.165
}
```

Conclusion: the in-kernel top-k loop was a major TPOT bottleneck. Top-8 remains
the fastest measured 32k setting, while top-16 is now close enough to be a
practical quality/speed tradeoff. Compared with default Quest top-64, the
current fused top-8 path is about 9.4x faster by TPOT: `714.101 / 75.629`.

## 32k Output Regression Check

The benchmark now records complete generated text and token ids, not only a
preview. A 32k default top-64 vs fused top-8 comparison was run with:

```bash
.venv-mlx/bin/python benchmarks/quest_sparse_decode_benchmark.py \
  --model mlx-community/Qwen3-4B-4bit-DWQ-053125 \
  --target-prompt-tokens 32768 \
  --page-budget 64 \
  --fused-page-budget 8 \
  --max-tokens 16 \
  --cases both \
  --output /private/tmp/alayajet_quest_sparse_bench/long32k_quality_default64_vs_fused8.jsonl
```

Comparison:

```text
case                    prompt_tokens  completion_tokens  tpot_ms  speedup_vs_first  text_match  token_match
default[budget=64]      33619          15                 616.65   1.0               True        True
fused_metal[budget=8]   33619          15                 71.936   8.572             True        True
```

Both paths generated the same token ids:

```json
[45861, 102285, 16744, 113272, 85106, 101940, 108260, 100768, 33108, 82707, 99982, 24360, 112920, 1773, 151645]
```

Generated text:

```text
长上下文推理需要减少注意力计算和KV缓存搬运。
```

## 32k Same Top-k TPOT

To compare TPOT with the sparse budget held constant, both paths were run with
top-8/page-budget 8:

```bash
.venv-mlx/bin/python benchmarks/quest_sparse_decode_benchmark.py \
  --model mlx-community/Qwen3-4B-4bit-DWQ-053125 \
  --target-prompt-tokens 32768 \
  --page-budget 8 \
  --fused-page-budget 8 \
  --max-tokens 16 \
  --cases both \
  --output /private/tmp/alayajet_quest_sparse_bench/long32k_same_top8_default_vs_fused.jsonl
```

Result:

```text
case                    prompt_tokens  completion_tokens  tpot_ms  speedup_vs_first  text_match  token_match
default[budget=8]       33619          15                 340.891  1.0               True        True
fused_metal[budget=8]   33619          15                 72.035   4.732             True        True
```

Both paths generated the same token ids and text as the default top-64
regression check above.

Additional same-top-k 32k runs were made for larger budgets:

```text
budget  default_tpot_ms  fused_tpot_ms  speedup  token_match
26      478.995          104.222        4.596x   True
32      446.121          115.603        3.859x   True
64      664.944          285.403        2.330x   True
```

All runs used:

```bash
.venv-mlx/bin/python benchmarks/quest_sparse_decode_benchmark.py \
  --model mlx-community/Qwen3-4B-4bit-DWQ-053125 \
  --target-prompt-tokens 32768 \
  --page-budget <K> \
  --fused-page-budget <K> \
  --max-tokens 16 \
  --cases both
```

The output token ids matched default Quest for each budget. The fused kernel
continues to win at larger top-k, but the speedup shrinks as the selected token
count grows.

## LongBench Accuracy Smoke

LongBench data/config were downloaded from the official sources and
`benchmarks/eval_longbench_all.py` was updated to pass `enable_thinking=False`
to Qwen3 chat templates. Without that, the model spends the short generation
budget in `<think>` text and the retrieval score is misleading.

This smoke uses `passage_retrieval_en`, first 3 samples, `max_tokens=32`:

```text
case            budget  samples  score  avg_tpot_ms  avg_ttft_s  avg_latency_s
default_top64   64      3        100.0  389.140      83.560      85.470
fused_top26     26      3        100.0  119.301      87.316      87.987
fused_top32     32      3        100.0  110.280      87.278      87.888
fused_top64     64      3        100.0  138.139      87.291      88.038
```

This is a small retrieval subset, not a full LongBench run. It verifies that the
Metal fused path preserved accuracy on the checked retrieval samples while
reducing decode TPOT.

## API Server Smoke

The experimental fused Metal sparse path can be enabled from the API server
entry point with:

```bash
.venv-mlx/bin/python -m alayajet.api_server \
  --model mlx-community/Qwen3-4B-4bit-DWQ-053125 \
  --host 127.0.0.1 \
  --port 18082 \
  --max-tokens 4 \
  --chat-template-args '{"enable_thinking":false}' \
  --quest-reset-cache \
  --page-budget 4 \
  --quest-metal-sparse \
  --quest-metal-page-budget 2 \
  --quest-timing \
  --quest-timing-sync
```

Smoke request:

```text
POST /v1/chat/completions
prompt_tokens: 1136
completion_tokens: 4
status: 200
```

Response preview:

```json
{
  "model": "mlx-community/Qwen3-4B-4bit-DWQ-053125",
  "usage": {
    "prompt_tokens": 1136,
    "completion_tokens": 4,
    "total_tokens": 1140
  },
  "choices": [
    {
      "finish_reason": "length",
      "message": {
        "content": "你提供的文本中"
      }
    }
  ]
}
```
