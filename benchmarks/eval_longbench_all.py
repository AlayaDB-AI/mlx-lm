import argparse
import json
import os
import sys
import time
import mlx.core as mx
from mlx_lm import load, generate, stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alayajet.engine import AlayaEngine
from alayajet.features.quest.integration import QuestFeature

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_jsonl(file_path, num_samples, skip_samples=0):
    yielded = 0
    skipped = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            if skipped < skip_samples:
                skipped += 1
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                preview = line[:120].rstrip("\n")
                raise RuntimeError(
                    f"Failed to parse JSONL at {file_path}:{line_no}. "
                    f"First line preview: {preview!r}."
                ) from e
            yield item
            yielded += 1
            if num_samples is not None and num_samples > 0 and yielded >= num_samples:
                break

def count_jsonl_rows(file_path):
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count

def infer_dataset_from_path(file_path):
    base = os.path.basename(file_path)
    if base.endswith(".jsonl"):
        return base[:-6]
    return os.path.splitext(base)[0]

def parse_datasets(value):
    if not value:
        return []
    if value.strip().lower() == "all":
        return ["all"]
    return [item.strip() for item in value.split(",") if item.strip()]

NO_CHAT_TEMPLATE_DATASETS = {
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
}

def sanitize_model_name(model_name):
    return model_name.replace("/", "_").replace(" ", "_")

def reset_quest_state(engine):
    if engine is None:
        return
    for feature in engine.features:
        if isinstance(feature, QuestFeature) and feature.controller is not None:
            feature.controller.clean_states()
            feature._decode_step = 0
            break
    mx.clear_cache()

def main():
    parser = argparse.ArgumentParser(description="Lite Evaluation on LongBench (official prompts/maxlen)")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-7B-Instruct-1M-4bit", help="Model path")
    parser.add_argument("--quest", action="store_true", help="Enable Quest Feature")
    parser.add_argument("--page-budget", type=int, default=128, help="Quest page budget")
    parser.add_argument("--page-size", type=int, default=64, help="Quest page size in tokens")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate (<=0 means all)",
    )
    parser.add_argument("--data-path", type=str, default="", help="Path to a single LongBench <dataset>.jsonl")
    parser.add_argument("--data-dir", type=str, default="benchmarks/data/LongBench", help="Directory with LongBench *.jsonl files")
    parser.add_argument("--dataset", type=str, default="", help="Single dataset name (overrides --datasets)")
    parser.add_argument("--datasets", type=str, default="narrativeqa", help="Comma-separated dataset names or 'all'")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="3rdparty/Quest/evaluation/LongBench/config",
        help="LongBench config dir with dataset2prompt.json + dataset2maxlen.json",
    )
    parser.add_argument("--cache-dir", type=str, default="./kv_cache_eval", help="Quest cache directory")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max tokens for generation (otherwise use dataset2maxlen)",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing outputs (default; skips completed samples)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start fresh and overwrite existing outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pred",
        help="Directory to write predictions",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="",
        help="Subdirectory name for outputs (defaults to sanitized model name)",
    )
    parser.add_argument("--prompt-repeat", type=int, default=1, help="Repeat each prompt N times to extend context")
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Truncate the prompt to the last N tokens before generation"
    )
    parser.add_argument("--timing", action="store_true", help="Enable baseline timing breakdown")
    parser.add_argument(
        "--timing-sync",
        dest="timing_sync",
        action="store_true",
        default=True,
        help="Sync MLX ops for accurate timing (default)"
    )
    parser.add_argument(
        "--timing-no-sync",
        dest="timing_sync",
        action="store_false",
        help="Do not sync MLX ops (lower overhead, less accurate)"
    )
    parser.add_argument("--quest-timing", action="store_true", help="Enable Quest timing breakdown")
    parser.add_argument(
        "--quest-timing-sync",
        dest="quest_timing_sync",
        action="store_true",
        default=True,
        help="Sync MLX ops for accurate timing (default)"
    )
    parser.add_argument(
        "--quest-timing-no-sync",
        dest="quest_timing_sync",
        action="store_false",
        help="Do not sync MLX ops (lower overhead, less accurate)"
    )
    parser.add_argument(
        "--quest-trace-output",
        type=str,
        default="",
        help="Write Quest timing trace (e.g. timeline.svg or trace.json)"
    )
    parser.add_argument(
        "--quest-trace-decode-steps",
        type=int,
        default=3,
        help="Number of decode steps to include in Quest trace (<=0 disables decode tracing)"
    )
    parser.add_argument(
        "--quest-prefill-layer-log",
        action="store_true",
        help="Log per-layer prefill time for Quest"
    )
    parser.add_argument(
        "--quest-prefill-io-log",
        action="store_true",
        help="Log per-chunk prefill read/attn overlap for Quest"
    )
    parser.add_argument(
        "--quest-lru-log",
        action="store_true",
        help="Log per-step LRU hit rate during Quest decode"
    )
    parser.add_argument(
        "--quest-mem-log",
        action="store_true",
        help="Log Quest memory checkpoints during prefill"
    )
    parser.add_argument(
        "--quest-disable-os-cache",
        dest="quest_disable_os_cache",
        action="store_true",
        help="Disable OS page cache for Quest KV backing file (default)"
    )
    parser.add_argument(
        "--quest-enable-os-cache",
        dest="quest_disable_os_cache",
        action="store_false",
        help="Allow OS page cache for Quest KV backing file"
    )
    parser.add_argument(
        "--quest-reset-between-samples",
        dest="quest_reset_between_samples",
        action="store_true",
        default=True,
        help="Reset Quest cache between samples (default)",
    )
    parser.add_argument(
        "--no-quest-reset-between-samples",
        dest="quest_reset_between_samples",
        action="store_false",
        help="Keep Quest cache across samples (not recommended for benchmarks)",
    )
    parser.set_defaults(quest_disable_os_cache=True)
    args = parser.parse_args()

    config_dir = args.config_dir
    dataset2prompt_path = os.path.join(config_dir, "dataset2prompt.json")
    dataset2maxlen_path = os.path.join(config_dir, "dataset2maxlen.json")
    if not os.path.exists(dataset2prompt_path) or not os.path.exists(dataset2maxlen_path):
        print(f"Error: LongBench config files not found in {config_dir}.")
        return
    dataset2prompt = load_json(dataset2prompt_path)
    dataset2maxlen = load_json(dataset2maxlen_path)

    datasets = []
    data_paths = {}
    if args.data_path:
        if not os.path.exists(args.data_path):
            print(f"Error: Data file {args.data_path} not found.")
            return
        dataset_name = args.dataset.strip() if args.dataset else infer_dataset_from_path(args.data_path)
        datasets = [dataset_name]
        data_paths[dataset_name] = args.data_path
    else:
        if args.dataset.strip():
            datasets = [args.dataset.strip()]
        else:
            parsed = parse_datasets(args.datasets)
            if parsed == ["all"]:
                datasets = sorted(dataset2prompt.keys())
            else:
                datasets = parsed
        for dataset_name in datasets:
            data_paths[dataset_name] = os.path.join(args.data_dir, f"{dataset_name}.jsonl")

    output_name = args.output_name.strip() or sanitize_model_name(args.model)
    output_dir = os.path.join(args.output_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- LongBench Lite Eval ---")
    print(f"Model: {args.model}")
    print(f"Quest: {args.quest} (Budget: {args.page_budget})")
    print(f"Samples: {args.num_samples}")
    print(f"Datasets: {', '.join(datasets) if datasets else '(none)'}")
    print(f"Output: {output_dir}")
    if args.quest and args.timing and not args.quest_timing:
        print("[Timing] Baseline timing is ignored with --quest; use --quest-timing for Quest breakdown.")

    # 1. Load Model & Engine
    print("Loading model...")
    model, tokenizer = load(args.model)
    
    engine = None
    if args.quest:
        print(f"Attaching Quest Engine (with Chunking)...")
        if os.path.exists(args.cache_dir):
            import shutil
            shutil.rmtree(args.cache_dir)
        
        engine = AlayaEngine.with_quest(
            page_budget=args.page_budget, 
            page_size=args.page_size,
            cache_dir=args.cache_dir,
            async_disk_write=True,
            timing=args.quest_timing,
            timing_sync=args.quest_timing_sync,
            trace_output=args.quest_trace_output or None,
            trace_decode_steps=args.quest_trace_decode_steps,
            log_prefill_layer_timing=args.quest_prefill_layer_log,
            log_prefill_io_overlap=args.quest_prefill_io_log,
            log_decode_lru_hit_rate=args.quest_lru_log,
            log_memory=args.quest_mem_log,
            log_memory_sync=args.quest_timing_sync,
            log_prefill_progress=False,
            disable_os_cache=args.quest_disable_os_cache,
        )
        
        # from alayajet.features.chunking import ChunkComputationFeature
        # engine.add_feature(ChunkComputationFeature(chunk_size=2048))
        
        engine.attach(model)
    elif args.timing:
        from alayajet.features.timing import TimingFeature
        engine = AlayaEngine()
        engine.add_feature(TimingFeature(timing_sync=args.timing_sync))
        engine.attach(model)

    # 2. Load Data + Run
    latency_sum = 0.0
    latency_count = 0
    dataset_outputs = {}
    for dataset_name in datasets:
        dataset_file = os.path.join(output_dir, f"{dataset_name}.jsonl")
        if os.path.exists(dataset_file) and not args.resume:
            os.remove(dataset_file)
        dataset_outputs[dataset_name] = dataset_file
    for dataset_name in datasets:
        data_path = data_paths.get(dataset_name, "")
        if not data_path or not os.path.exists(data_path):
            print(f"[WARN] Data file missing for {dataset_name}: {data_path}")
            continue
        base_name = dataset_name[:-2] if dataset_name.endswith("_e") else dataset_name
        prompt_format = dataset2prompt.get(dataset_name)
        if not prompt_format:
            if dataset_name.endswith("_e"):
                prompt_format = dataset2prompt.get(base_name)
            if not prompt_format:
                print(f"[WARN] dataset2prompt missing for {dataset_name}, skipping.")
                continue

        max_tokens = args.max_tokens
        if max_tokens is None:
            max_tokens = dataset2maxlen.get(dataset_name)
            if max_tokens is None and dataset_name.endswith("_e"):
                max_tokens = dataset2maxlen.get(dataset_name[:-2])
        if max_tokens is None:
            print(f"[WARN] dataset2maxlen missing for {dataset_name}; using 32.")
            max_tokens = 32

        skip_samples = 0
        if args.resume and os.path.exists(dataset_outputs[dataset_name]):
            skip_samples = count_jsonl_rows(dataset_outputs[dataset_name])

        effective_num_samples = args.num_samples
        if args.num_samples is not None and args.num_samples > 0:
            effective_num_samples = max(args.num_samples - skip_samples, 0)

        if effective_num_samples == 0 and skip_samples > 0:
            print(f"[INFO] {dataset_name}: resume found {skip_samples} samples, skipping generation.")
            continue

        samples = iter_jsonl(data_path, effective_num_samples, skip_samples=skip_samples)
        display_total = args.num_samples if args.num_samples is not None and args.num_samples > 0 else "all"
        for idx, sample in enumerate(samples, start=skip_samples + 1):
            print(f"\n[{dataset_name}] [{idx}/{display_total}]")

            prompt = prompt_format.format(context=sample["context"], input=sample["input"])
            if args.prompt_repeat > 1:
                prompt = "\n\n".join([prompt] * args.prompt_repeat)

            # Match LongBench reference behavior: skip chat template for specific tasks.
            if hasattr(tokenizer, "apply_chat_template") and base_name not in NO_CHAT_TEMPLATE_DATASETS:
                messages = [{"role": "user", "content": prompt}]
                prompt_formatted = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            else:
                prompt_formatted = prompt

            wrapped_tokenizer = tokenizer
            if not isinstance(wrapped_tokenizer, TokenizerWrapper):
                wrapped_tokenizer = TokenizerWrapper(tokenizer)
            add_special_tokens = (
                wrapped_tokenizer.bos_token is None
                or not prompt_formatted.startswith(wrapped_tokenizer.bos_token)
            )
            prompt_tokens = wrapped_tokenizer.encode(
                prompt_formatted, add_special_tokens=add_special_tokens
            )
            if args.max_prompt_tokens is not None and len(prompt_tokens) > args.max_prompt_tokens:
                prompt_tokens = prompt_tokens[-args.max_prompt_tokens :]
                print(f"[DEBUG] Prompt truncated to last {args.max_prompt_tokens} tokens.")
            token_count = len(prompt_tokens)
            print(f"Context Length: {sample.get('length')} | Prompt Tokens: {token_count}")

            start_time = time.time()

            # Generation
            prefill_step_size = 8192
            print(f"[DEBUG] Quest: {args.quest}, Prefill Step Size: {prefill_step_size}")

            ttft = None
            tpot = None
            decode_time = None
            prediction = ""
            last_response = None
            progress_callback = None
            for response in stream_generate(
                model,
                tokenizer,
                prompt=prompt_tokens,
                max_tokens=max_tokens,
                prefill_step_size=prefill_step_size,
                prompt_progress_callback=progress_callback,
            ):
                if ttft is None and response.prompt_tps:
                    ttft = response.prompt_tokens / response.prompt_tps
                last_response = response
                prediction += response.text
            if last_response and last_response.generation_tps:
                tpot = 1.0 / last_response.generation_tps
                if last_response.generation_tokens:
                    decode_time = (
                        last_response.generation_tokens / last_response.generation_tps
                    )

            duration = time.time() - start_time
            prediction = prediction.strip()

            print(f"Question: {sample['input']}")
            print(f"Prediction: {prediction}")
            print(f"Ground Truth: {sample['answers']}")
            if ttft is not None:
                print(f"TTFT: {ttft:.2f}s")
            if tpot is not None:
                print(f"TPOT: {tpot*1000:.2f}ms")
            if decode_time is not None:
                print(f"Decode Time: {decode_time:.2f}s")
            print(f"Latency: {duration:.2f}s")

            output_payload = {
                "pred": prediction,
                "answers": sample.get("answers", []),
                "all_classes": sample.get("all_classes", []),
                "length": sample.get("length", None),
                "dataset": sample.get("dataset", dataset_name),
                "input": sample.get("input", ""),
                "latency": duration,
                "ttft": ttft,
                "tpot": tpot,
                "decode_time": decode_time,
                "tokens": token_count,
            }
            with open(dataset_outputs[dataset_name], "a", encoding="utf-8") as f:
                f.write(json.dumps(output_payload, ensure_ascii=False) + "\n")

            latency_sum += duration
            latency_count += 1

            if args.quest and args.quest_reset_between_samples:
                reset_quest_state(engine)

    # 3. Summary
    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    if latency_count:
        avg_latency = latency_sum / latency_count
        print(f"Average Latency: {avg_latency:.2f}s")
    else:
        print("No results (missing data files or configs).")

    # Detach
    if engine:
        engine.detach()

if __name__ == "__main__":
    main()
