"""Compare Quest default decode against experimental Metal sparse paths.

Example:

    .venv-mlx/bin/python benchmarks/quest_sparse_decode_benchmark.py \
        --model mlx-community/Qwen3-4B-4bit-DWQ-053125 \
        --target-prompt-tokens 32768 --page-budget 64 --max-tokens 64

The script uses the in-process AlayaJet engine instead of the HTTP server so it
can reliably detach the engine and print Quest timing summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
import mlx_lm.models.cache as mlx_cache

from alayajet.engine import AlayaEngine
from alayajet.api_server.quest_kv_cache import QuestDiskCache


def build_prompt(args, tokenizer) -> tuple[str, int, int, list[int]]:
    def make_prompt(repeat_factor: int) -> tuple[str, int, list[int]]:
        prompt_text = args.prompt_prefix + (args.prompt_unit * repeat_factor)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = tokenizer.encode(prompt)
        return prompt, len(tokens), tokens

    repeat_factor = max(1, args.repeat_factor)
    if args.target_prompt_tokens:
        low = 1
        high = repeat_factor
        prompt, token_count, tokens = make_prompt(high)
        while token_count < args.target_prompt_tokens:
            low = high + 1
            high *= 2
            prompt, token_count, tokens = make_prompt(high)

        best_prompt = prompt
        best_tokens = token_count
        best_token_ids = tokens
        best_repeat = high
        while low <= high:
            mid = (low + high) // 2
            prompt, token_count, tokens = make_prompt(mid)
            if token_count >= args.target_prompt_tokens:
                best_prompt = prompt
                best_tokens = token_count
                best_token_ids = tokens
                best_repeat = mid
                high = mid - 1
            else:
                low = mid + 1
        return best_prompt, best_tokens, best_repeat, best_token_ids

    prompt, token_count, tokens = make_prompt(repeat_factor)
    return prompt, token_count, repeat_factor, tokens


def _cache_id(args, prompt_tokens: list[int]) -> str:
    model_slug = args.model.replace("/", "_").replace(":", "_")
    digest = hashlib.sha1(
        ",".join(str(int(t)) for t in prompt_tokens).encode("utf-8")
    ).hexdigest()[:12]
    return (
        f"{model_slug}_p{args.page_size}_max{args.quest_max_seq_len or 0}_"
        f"n{len(prompt_tokens)}_{digest}"
    )


def _set_feature_max_seq_len(engine, max_seq_len: int):
    if not max_seq_len:
        return
    for feature in engine.features:
        if hasattr(feature, "max_seq_len"):
            feature.max_seq_len = max_seq_len


def _make_quest_engine(args, cache_dir: Path, *, timing: bool) -> AlayaEngine:
    return AlayaEngine.with_quest(
        page_budget=args.page_budget,
        cache_dir=str(cache_dir),
        timing=timing,
        timing_sync=True,
        trace_decode_steps=args.trace_decode_steps,
        page_size=args.page_size,
        log_prefill_progress=False,
        log_decode_lru_hit_rate=args.quest_lru_log,
        log_memory=args.quest_mem_log,
        log_memory_sync=True,
    )


def _ensure_quest_prefix_cache(args, model, prompt: str, prompt_tokens: list[int], cache_dir: Path):
    state_path = cache_dir / "quest_state.json"
    backing_path = cache_dir / "kv_cache_pool.bin"
    if state_path.exists() and backing_path.exists():
        return

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    engine = _make_quest_engine(args, cache_dir, timing=False)
    _set_feature_max_seq_len(engine, args.quest_max_seq_len)
    try:
        engine.attach(model)
        _set_feature_max_seq_len(engine, args.quest_max_seq_len)
        token_mx = mx.array(prompt_tokens, dtype=mx.uint32)
        for start in range(0, len(prompt_tokens), args.prefill_step_size):
            chunk = token_mx[start : start + args.prefill_step_size]
            if chunk.size == 0:
                continue
            model(chunk[None])
            mx.eval(chunk)
            mx.clear_cache()
        controller = engine.features[0].controller
        QuestDiskCache().save(
            controller,
            str(cache_dir),
            prompt_tokens=prompt_tokens,
            model_id=args.model,
        )
    finally:
        engine.detach()


def _prepare_quest_working_cache(
    args,
    model,
    prompt: str,
    prompt_tokens: list[int],
    cache_name: str,
) -> tuple[Path, bool]:
    if not args.prefix_cache_root:
        return Path(args.cache_root) / cache_name, False

    prefix_root = Path(args.prefix_cache_root)
    if not prefix_root.is_absolute():
        prefix_root = Path.cwd() / prefix_root
    prefix_dir = prefix_root / "quest" / _cache_id(args, prompt_tokens)
    _ensure_quest_prefix_cache(args, model, prompt, prompt_tokens, prefix_dir)

    work_dir = Path(args.cache_root) / cache_name
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(prefix_dir, work_dir)
    return work_dir, True


def _ensure_dense_prompt_cache(args, model, prompt_tokens: list[int]):
    prefix_root = Path(args.prefix_cache_root)
    if not prefix_root.is_absolute():
        prefix_root = Path.cwd() / prefix_root
    cache_file = prefix_root / "dense" / f"{_cache_id(args, prompt_tokens)}.safetensors"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        return mlx_cache.load_prompt_cache(str(cache_file))

    prompt_cache = mlx_cache.make_prompt_cache(model)
    prefix_tokens = prompt_tokens[:-1]
    for start in range(0, len(prefix_tokens), args.prefill_step_size):
        chunk = prefix_tokens[start : start + args.prefill_step_size]
        if not chunk:
            continue
        model(mx.array(chunk, dtype=mx.uint32)[None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])
        mx.clear_cache()
    mlx_cache.save_prompt_cache(
        str(cache_file),
        prompt_cache,
        metadata={
            "model": args.model,
            "prompt_tokens": str(len(prompt_tokens)),
            "prefix_tokens": str(len(prefix_tokens)),
        },
    )
    return prompt_cache


def _suppress_eos_processors(tokenizer):
    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_ids is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        eos_ids = [] if eos_id is None else [eos_id]
    eos_ids = [int(x) for x in eos_ids if x is not None]
    if not eos_ids:
        return None
    eos_ids_mx = mx.array(eos_ids, dtype=mx.int32)

    def suppress_eos(_, logits):
        logits[:, eos_ids_mx] = -1e9
        return logits

    return [suppress_eos]


def run_case(args, *, case: str) -> dict:
    fused = case == "fused_metal"
    selected_metal = case == "selected_metal"
    resident_write = case == "selected_resident_write"
    selected_metal = selected_metal or resident_write
    dense = case == "dense"
    os.environ["ALAYAJET_QUEST_METAL_SPARSE"] = "1" if fused else "0"
    os.environ["ALAYAJET_QUEST_METAL_SELECTED_PAGE"] = "1" if selected_metal else "0"
    os.environ["ALAYAJET_QUEST_RESIDENT_FRAME_WRITE"] = "1" if resident_write else "0"
    if resident_write:
        os.environ["ALAYAJET_QUEST_METAL_SELECTED_FRAME"] = "resident"
    elif selected_metal and os.environ.get("ALAYAJET_QUEST_METAL_SELECTED_FRAME"):
        pass
    else:
        os.environ.pop("ALAYAJET_QUEST_METAL_SELECTED_FRAME", None)
    if fused and args.fused_page_budget:
        os.environ["ALAYAJET_QUEST_METAL_PAGE_BUDGET"] = str(args.fused_page_budget)
    else:
        os.environ.pop("ALAYAJET_QUEST_METAL_PAGE_BUDGET", None)
    model, tokenizer = load(args.model)
    args._tokenizer = tokenizer
    prompt, prompt_tokens_estimate, repeat_factor, prompt_token_ids = build_prompt(
        args, tokenizer
    )
    prompt_for_generation = prompt
    prompt_cache = None
    prefix_cache_hit = False
    engine = None
    if not dense:
        if fused:
            cache_name = "fused_metal"
        elif resident_write:
            cache_name = "selected_resident_write"
        elif selected_metal:
            cache_name = "selected_metal"
        else:
            cache_name = "default"
        cache_dir, prefix_cache_hit = _prepare_quest_working_cache(
            args,
            model,
            prompt,
            prompt_token_ids,
            cache_name,
        )
        engine = _make_quest_engine(args, cache_dir, timing=True)
        _set_feature_max_seq_len(engine, args.quest_max_seq_len)
        engine.attach(model)
        _set_feature_max_seq_len(engine, args.quest_max_seq_len)
        if prefix_cache_hit:
            controller = engine.features[0].ensure_controller(cache_dir=str(cache_dir))
            _, rest, _ = QuestDiskCache().load(
                controller,
                str(cache_dir),
                prompt_token_ids,
                model_id=args.model,
            )
            prompt_for_generation = rest
    elif args.prefix_cache_root:
        prompt_cache = _ensure_dense_prompt_cache(args, model, prompt_token_ids)
        prompt_for_generation = prompt_token_ids[-1:]
        prefix_cache_hit = True

    logits_processors = _suppress_eos_processors(tokenizer) if args.suppress_eos else None

    generated = []
    generated_token_ids = []
    token_times = []
    start = time.perf_counter()
    try:
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt_for_generation,
            max_tokens=args.max_tokens,
            prefill_step_size=args.prefill_step_size,
            prompt_cache=prompt_cache,
            logits_processors=logits_processors,
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
        "prompt_tokens": prompt_tokens_estimate,
        "measured_prompt_tokens": prompt_tokens if token_times else prompt_tokens_estimate,
        "completion_tokens": generation_tokens if token_times else 0,
        "page_size": args.page_size,
        "page_budget": args.page_budget,
        "fused_page_budget": args.fused_page_budget if fused else None,
        "selected_page_metal": selected_metal,
        "selected_frame_mode": os.environ.get("ALAYAJET_QUEST_METAL_SELECTED_FRAME"),
        "resident_frame_write": resident_write,
        "prefill_step_size": args.prefill_step_size,
        "max_tokens": args.max_tokens,
        "repeat_factor": repeat_factor,
        "prompt_chars": len(prompt),
        "prefix_cache_hit": prefix_cache_hit,
        "prefix_cache_root": args.prefix_cache_root,
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids,
        "generated_preview": generated_text[:80],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-4bit-DWQ-053125")
    parser.add_argument("--page-budget", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=64)
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
    parser.add_argument(
        "--quest-lru-log",
        action="store_true",
        help="Log per-step LRU and selected-page hit rate during Quest decode.",
    )
    parser.add_argument(
        "--quest-mem-log",
        action="store_true",
        help="Log Quest memory checkpoints during prefill/decode.",
    )
    parser.add_argument("--cache-root", default="tpot_context_sweep_results/cache")
    parser.add_argument("--prefix-cache-root", default="")
    parser.add_argument("--suppress-eos", action="store_true")
    parser.add_argument("--output", default="tpot_context_sweep_results/results.jsonl")
    parser.add_argument(
        "--cases",
        choices=(
            "default",
            "fused",
            "selected",
            "resident_write",
            "dense",
            "both",
            "metal",
            "all",
        ),
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
    elif args.cases == "selected":
        cases = ["selected_metal"]
    elif args.cases == "resident_write":
        cases = ["selected_resident_write"]
    elif args.cases == "dense":
        cases = ["dense"]
    elif args.cases == "metal":
        cases = ["fused_metal", "selected_metal"]
    elif args.cases == "all":
        cases = ["dense", "default", "fused_metal", "selected_metal", "selected_resident_write"]
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
