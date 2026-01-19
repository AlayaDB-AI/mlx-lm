
import mlx.core as mx
from mlx_lm import load, generate
import sys
import os
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alayajet import AlayaEngine
from alayajet.features.monitor import VRAMMonitorFeature
from alayajet.features.chunking import ChunkComputationFeature

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_len", type=int, default=10000, help="Prompt words")
    parser.add_argument("--chunk", action="store_true", help="Enable AlayaJet Chunking")
    parser.add_argument("--chunk_size", type=int, default=2048, help="Chunk size")
    args = parser.parse_args()

    print(f"=== AlayaJet Chunk-Only Comparison ===")
    print(f"Prompt: {args.prompt_len} words")
    print(f"AlayaJet Chunking: {args.chunk} (Size: {args.chunk_size})")
    
    model_path = "mlx-community/Qwen2.5-7B-Instruct-1M-4bit"
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    
    engine = AlayaEngine()
    engine.add_feature(VRAMMonitorFeature(interval=4))
    
    if args.chunk:
        engine.add_feature(ChunkComputationFeature(chunk_size=args.chunk_size))
        
    engine.attach(model)
    
    prompt = "word " * args.prompt_len
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        add_generation_prompt=True
    )
    
    print("\nStarting generation (NATIVE CHUNKING DISABLED)...")
    
    # We disable native chunking to force model to process all tokens at once (if possible)
    # This will truly test if our Feature can split the big task inside the model.
    try:
        generate(
            model, 
            tokenizer, 
            prompt=formatted_prompt, 
            max_tokens=1, 
            verbose=True,
            prefill_step_size=1000000 
        )
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
    
    engine.detach()

if __name__ == "__main__":
    main()
