
import mlx.core as mx
from mlx_lm import load, generate
import sys
import os

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alayajet import AlayaEngine

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_len", type=int, default=100, help="Approximate length of prompt in words")
    args = parser.parse_args()

    print("=== AlayaJet Monitor Only Test ===")
    
    model_path = "mlx-community/Qwen2.5-7B-Instruct-1M-4bit"
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    
    # Initialize Engine with Monitor ONLY
    engine = AlayaEngine.with_monitor(interval=4)
    engine.attach(model)
    
    # Generate long dummy prompt
    prompt = "word " * args.prompt_len
    
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        add_generation_prompt=True
    )
    
    print(f"\nStarting generation (Prompt Len ~{args.prompt_len} words)...")
    generate(model, tokenizer, prompt=formatted_prompt, max_tokens=5, verbose=True)
    
    engine.detach()

if __name__ == "__main__":
    main()
