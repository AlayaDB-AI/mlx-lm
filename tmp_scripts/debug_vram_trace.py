import sys
import os
import mlx.core as mx
from mlx_lm import load, generate
import linecache

# Global stats
max_vram_seen = 0
max_vram_location = None
max_vram_code = None

def memory_trace(frame, event, arg):
    if event != 'line':
        return memory_trace

    global max_vram_seen, max_vram_location, max_vram_code
    
    # Check Active VRAM at this exact line execution
    active = mx.get_active_memory()
    
    if active > max_vram_seen:
        max_vram_seen = active
        lineno = frame.f_lineno
        filename = frame.f_code.co_filename
        # Only track files in mlx_lm to reduce noise, unless you want system-wide
        if "mlx_lm" in filename:
            max_vram_location = f"{filename}:{lineno}"
            max_vram_code = linecache.getline(filename, lineno).strip()
    
    return memory_trace

def main():
    print("=== Peak VRAM Tracer ===")
    print("Tracking where Peak Memory occurs...")
    
    model_path = "mlx-community/Qwen2.5-7B-Instruct-1M-4bit"
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    
    # 30k tokens should be enough to cause a visible peak without crashing
    prompt = "word " * 10000 
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        add_generation_prompt=True
    )
    
    print("Starting generation with trace...")
    
    sys.settrace(memory_trace)
    
    try:
        # Just generate 1 token to trigger prefill
        generate(model, tokenizer, prompt=formatted_prompt, max_tokens=1, verbose=False)
    except Exception as e:
        print(f"Error during run: {e}")
    finally:
        sys.settrace(None)
        
        print("\n" + "="*50)
        print("🏆 PEAK MEMORY LOCATION DETECTED")
        print("="*50)
        print(f"Peak VRAM: {max_vram_seen / 1024**3:.4f} GB")
        if max_vram_location:
            print(f"Location:  {max_vram_location}")
            print(f"Code Line: {max_vram_code}")
        else:
            print("Location:  (Outside of mlx_lm or not captured)")
        print("="*50)

if __name__ == "__main__":
    main()