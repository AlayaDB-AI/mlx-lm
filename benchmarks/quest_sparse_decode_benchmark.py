"""Compare Quest default decode against the experimental fused Metal sparse path.

Example:

    .venv-mlx/bin/python benchmarks/quest_sparse_decode_benchmark.py \
        --model mlx-community/Qwen3-4B-4bit-DWQ-053125 \
        --target-prompt-tokens 32768 --page-budget 64 --max-tokens 64

The script uses the in-process AlayaJet engine instead of the HTTP server so it
can reliably detach the engine and print Quest timing summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from mlx_lm import load, stream_generate

from alayajet.engine import AlayaEngine


def build_prompt(args, tokenizer) -> tuple[str, int, int]:
    def make_prompt(repeat_factor: int) -> tuple[str, int]:
        prompt_text = args.prompt_prefix + (args.prompt_unit * repeat_factor)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return prompt, len(tokenizer.encode(prompt))

    repeat_factor = max(1, args.repeat_factor)
    if args.target_prompt_tokens:
        low = 1
        high = repeat_factor
        prompt, token_count = make_prompt(high)
        while token_count < args.target_prompt_tokens:
            low = high + 1
            high *= 2
            prompt, token_count = make_prompt(high)

        best_prompt = prompt
        best_tokens = token_count
        best_repeat = high
        while low <= high:
            mid = (low + high) // 2
            prompt, token_count = make_prompt(mid)
            if token_count >= args.target_prompt_tokens:
                best_prompt = prompt
                best_tokens = token_count
                best_repeat = mid
                high = mid - 1
            else:
                low = mid + 1
        return best_prompt, best_tokens, best_repeat

    prompt, token_count = make_prompt(repeat_factor)
    return prompt, token_count, repeat_factor


def run_case(args, *, case: str) -> dict:
    fused = case == "fused_metal"
    dense = case == "dense"
    os.environ["ALAYAJET_QUEST_METAL_SPARSE"] = "1" if fused else "0"
    if fused and args.fused_page_budget:
        os.environ["ALAYAJET_QUEST_METAL_PAGE_BUDGET"] = str(args.fused_page_budget)
    else:
        os.environ.pop("ALAYAJET_QUEST_METAL_PAGE_BUDGET", None)
    model, tokenizer = load(args.model)
    engine = None
    if not dense:
        cache_dir = Path(args.cache_root) / ("fused_metal" if fused else "default")
        engine = AlayaEngine.with_quest(
            page_budget=args.page_budget,
            cache_dir=str(cache_dir),
            timing=True,
            timing_sync=True,
            trace_decode_steps=args.trace_decode_steps,
            log_prefill_progress=False,
        )
        if args.quest_max_seq_len:
            for feature in engine.features:
                if hasattr(feature, "max_seq_len"):
                    feature.max_seq_len = args.quest_max_seq_len
        engine.attach(model)
        if args.quest_max_seq_len:
            for feature in engine.features:
                if hasattr(feature, "max_seq_len"):
                    feature.max_seq_len = args.quest_max_seq_len
    prompt, prompt_tokens_estimate, repeat_factor = build_prompt(args, tokenizer)

    generated = []
    generated_token_ids = []
    token_times = []
    start = time.perf_counter()
    try:
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            prefill_step_size=args.prefill_step_size,
        ):
            generated.append(response.text)
            generated_token_ids.append(int(response.token))
            token_times.append(time.perf_counter())
            prompt_tokens = response.prompt_tokens
            prompt_tps = response.prompt_tps
            generation_tokens = response.generation_tokens
            generation_tps = response.generation_tps
    finally:
        elapsed = time.perf_counter() - start
        if engine is not None:
            engine.detach()

    ttft_s = token_times[0] - start if token_times else None
    if len(token_times) > 1:
        inter_token_ms = [
            (b - a) * 1000.0 for a, b in zip(token_times, token_times[1:])
        ]
        tpot_ms = sum(inter_token_ms) / len(inter_token_ms)
    else:
        inter_token_ms = []
        tpot_ms = None

    generated_text = "".join(generated)
    return {
        "case": case,
        "elapsed_s": round(elapsed, 4),
        "ttft_s": round(ttft_s, 4) if ttft_s is not None else None,
        "tpot_ms": round(tpot_ms, 3) if tpot_ms is not None else None,
        "inter_token_ms": [round(x, 3) for x in inter_token_ms],
        "prompt_tps": round(prompt_tps, 3) if token_times else None,
        "generation_tps": round(generation_tps, 3) if token_times else None,
        "prompt_tokens": prompt_tokens if token_times else prompt_tokens_estimate,
        "completion_tokens": generation_tokens if token_times else 0,
        "page_budget": args.page_budget,
        "fused_page_budget": args.fused_page_budget if fused else None,
        "prefill_step_size": args.prefill_step_size,
        "max_tokens": args.max_tokens,
        "repeat_factor": repeat_factor,
        "prompt_chars": len(prompt),
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids,
        "generated_preview": generated_text[:80],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-4bit-DWQ-053125")
    parser.add_argument("--page-budget", type=int, default=64)
    parser.add_argument(
        "--fused-page-budget",
        type=int,
        default=0,
        help="Optional approximate top-k budget used only by the fused Metal path.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--quest-max-seq-len",
        type=int,
        default=0,
        help="Optional Quest KV cache max sequence length override.",
    )
    parser.add_argument("--repeat-factor", type=int, default=120)
    parser.add_argument("--target-prompt-tokens", type=int, default=0)
    parser.add_argument("--trace-decode-steps", type=int, default=2)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--cache-root", default="tpot_context_sweep_results/cache")
    parser.add_argument("--output", default="tpot_context_sweep_results/results.jsonl")
    parser.add_argument(
        "--cases",
        choices=("default", "fused", "dense", "both", "all"),
        default="both",
        help="Which path to benchmark.",
    )
    parser.add_argument(
        "--prompt-prefix",
        default="请用一句话概括下面内容：",
    )
    parser.add_argument(
        "--prompt-unit",
        default="长上下文推理需要减少注意力计算和KV缓存搬运。",
    )
    args = parser.parse_args()

    if args.cases == "default":
        cases = ["default"]
    elif args.cases == "fused":
        cases = ["fused_metal"]
    elif args.cases == "dense":
        cases = ["dense"]
    elif args.cases == "all":
        cases = ["dense", "default", "fused_metal"]
    else:
        cases = ["default", "fused_metal"]
    results = [run_case(args, case=case) for case in cases]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    for row in results:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
