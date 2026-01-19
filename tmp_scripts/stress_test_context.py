
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models import cache as cache_utils
import sys
import os
import time
import argparse
import gc

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alayajet import AlayaEngine

def make_prompt(length_tokens):
    # Construct a dummy prompt of approximate length
    return "word " * length_tokens

def stress_test(args):
    print(f"=== Context Stress Test (Offload: {args.offload}) ===")
    
    # 1. Load Model
    model_path = args.model
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    
    # 2. Attach Engine if needed
    engine = None
    if args.offload:
        # Use a dedicated dir for stress test
        cache_dir = "./kv_stress_test"
        engine = AlayaEngine.with_offload(cache_dir=cache_dir)
        engine.attach(model)
        print(f">> AlayaJet Attached (Dir: {cache_dir})")

    # 3. Loop
    # We simulate a conversation history growing
    # Instead of generate() which might reset cache, we need to manage cache manually 
    # OR we just feed longer and longer prompts to generate() which re-prefills every time.
    # Re-prefilling (Prompt Caching disabled) is the TOUGHEST test for Prefill Memory.
    # If we want to test KV Cache Capacity (Decode OOM), we should use a persistent cache.
    
    # Let's test "Max Context Length" support. 
    # MLX generate() by default creates a new cache every call unless prompt_cache is passed.
    # So calling generate() with longer prompt tests PREFILL memory spike.
    # This is exactly what AlayaJet optimizes.
    
    current_len = args.start_len
    step = args.step
    max_len = args.max_len
    
    try:
        while current_len <= max_len:
            # --- VRAM Watchdog ---
            if args.vram_limit_gb > 0:
                # Force a sync to get accurate reading
                mx.eval(model.parameters()) 
                active_bytes = mx.metal.get_active_memory()
                limit_bytes = args.vram_limit_gb * 1024**3
                
                print(f"[Mem] Active: {active_bytes / 1024**3:.2f} GB / Limit: {args.vram_limit_gb:.2f} GB")
                
                if active_bytes > limit_bytes:
                    raise MemoryError(f"SIMULATED OOM: MLX used {active_bytes / 1024**3:.2f} GB, exceeding limit of {args.vram_limit_gb} GB")
            # ---------------------

            print(f"\n[Test] Length: {current_len} tokens")
            
            prompt_str = make_prompt(current_len)
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_str}], 
                add_generation_prompt=True
            )
            
            # Measure Time
            t0 = time.time()
            
            # We only generate a few tokens to verify it works, the heavy part is processing the prompt
            response = generate(
                model, 
                tokenizer, 
                prompt=formatted_prompt, 
                max_tokens=5, 
                verbose=False
            )
            
            t1 = time.time() 
            elapsed = t1 - t0
            print(f"  -> Success! Time: {elapsed:.2f}s")
            
            # Simple heuristic for "Too Slow / Throttling"
            if elapsed > 60: 
                print("  -> WARNING: Extremely slow generation detected (Swapping?).")
            
            current_len += step
            
            # Cleanup to avoid lingering memory from previous run if any
            if not args.offload:
                mx.metal.clear_cache()
                gc.collect()
                
    except Exception as e:
        print(f"\n❌ CRASHED at length ~{current_len}")
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\n⏹️ Test stopped by user.")
    
    if engine:
        engine.detach()
        # Clean up files
        # engine.offload_manager.reset_storage()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-0.5B-Instruct-4bit", help="Model path")
    parser.add_argument("--offload", action="store_true", help="Enable AlayaJet Offload")
    parser.add_argument("--start_len", type=int, default=1000, help="Starting context length")
    parser.add_argument("--step", type=int, default=1000, help="Step size")
    parser.add_argument("--max_len", type=int, default=32000, help="Max length limit")
    parser.add_argument("--vram_limit_gb", type=float, default=0, help="Soft VRAM limit in GB (0 to disable). Simulates Hard OOM.")
    args = parser.parse_args()
    
    stress_test(args)
