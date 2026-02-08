import argparse
import json
import time
import os
import mlx.core as mx
from mlx_lm import load, generate, stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper
from alayajet.engine import AlayaEngine

# LongBench NarrativeQA Prompt Template
PROMPT_TEMPLATE = (
    "You are given a story, which is a long document. Please answer the question based on the story.\n\n"
    "Story:\n"
    "{context}\n\n"
    "Question: {input}\n\n"
    "Answer:"
)

def load_jsonl(file_path, num_samples):
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if len(samples) >= num_samples:
                break
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                preview = line[:120].rstrip("\n")
                raise RuntimeError(
                    f"Failed to parse JSONL at {file_path}:{line_no}. "
                    f"First line preview: {preview!r}. "
                    "This usually means the file is not the expected LongBench *.jsonl "
                    "(e.g. it contains an error page like 'Entry not found')."
                ) from e
    return samples

def main():
    parser = argparse.ArgumentParser(description="Lite Evaluation on LongBench NarrativeQA")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-7B-Instruct-1M-4bit", help="Model path")
    parser.add_argument("--quest", action="store_true", help="Enable Quest Feature")
    parser.add_argument("--page-budget", type=int, default=128, help="Quest page budget")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to evaluate")
    parser.add_argument("--data-path", type=str, default="benchmarks/data/LongBench/narrativeqa.jsonl", help="Path to narrativeqa.jsonl")
    parser.add_argument("--cache-dir", type=str, default="./kv_cache_eval", help="Quest cache directory")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens for generation")
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
    parser.set_defaults(quest_disable_os_cache=True)
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data file {args.data_path} not found.")
        return

    print(f"--- LongBench Lite Eval ---")
    print(f"Model: {args.model}")
    print(f"Quest: {args.quest} (Budget: {args.page_budget})")
    print(f"Samples: {args.num_samples}")
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
            log_prefill_progress=True,
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

    # 2. Load Data
    samples = load_jsonl(args.data_path, args.num_samples)
    
    results = []
    
    for idx, sample in enumerate(samples):
        print(f"\n[{idx+1}/{args.num_samples}] Task: {sample.get('dataset', 'narrativeqa')}")
        
        prompt = PROMPT_TEMPLATE.format(context=sample['context'], input=sample['input'])
        if args.prompt_repeat > 1:
            prompt = "\n\n".join([prompt] * args.prompt_repeat)
        
        # Qwen Chat Template
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
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
            max_tokens=args.max_tokens,
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
        
        results.append({
            "input": sample['input'],
            "prediction": prediction,
            "answers": sample['answers'],
            "tokens": token_count,
            "latency": duration,
            "ttft": ttft,
            "tpot": tpot,
            "decode_time": decode_time
        })

    # 3. Summary
    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    avg_latency = sum(r['latency'] for r in results) / len(results)
    print(f"Average Latency: {avg_latency:.2f}s")
    
    # Detach
    if engine:
        engine.detach()

if __name__ == "__main__":
    main()
