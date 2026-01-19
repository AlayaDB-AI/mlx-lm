import mlx.core as mx
from mlx_lm import load, generate
import sys
import os
import time

# Ensure root directory is in python path to import alayajet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alayajet import AlayaEngine

def main():
    print("=== AlayaJet Engine Integration Test ===")
    
    # 1. Load Model
    model_path = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    
    # 2. Initialize Engine and Attach
    # engine = AlayaEngine(cache_dir="./kv_offload_test_dir") # Old API
    engine = AlayaEngine.with_offload(cache_dir="./kv_offload_test_dir") # New API
    engine.attach(model)
    
    # 3. Generate
    prompt = "Write a haiku about disk storage."
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    print("\nStarting generation...")
    start_time = time.time()
    
    response = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=15, verbose=True)
    
    end_time = time.time()
    print(f"\nGeneration finished in {end_time - start_time:.2f}s")
    
    # 4. Verify Disk Content
    # Access the OffloadFeature to get the manager
    offload_feature = engine.features[0]
    cache_dir = offload_feature.manager.cache_dir
    print(f"\nChecking cache directory: {cache_dir}")
    files = list(cache_dir.glob("*.safetensors"))
    if files:
        print(f"Success! Found {len(files)} offloaded layer files.")
        total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"Total size: {total_size:.2f} MB")
    else:
        print("FAILURE: No files found on disk.")

    # 5. Detach and Cleanup
    engine.detach()

if __name__ == "__main__":
    main()