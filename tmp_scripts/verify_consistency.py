import mlx.core as mx
from mlx_lm import load, generate
import sys
import os
import time
import argparse

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alayajet import AlayaEngine

def run_generation(model, tokenizer, prompt, use_alayajet=False):
    # Set seed for reproducibility
    mx.random.seed(42)
    
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        add_generation_prompt=True
    )
    
    engine = None
    if use_alayajet:
        engine = AlayaEngine.with_offload(cache_dir="./kv_consistency_test")
        engine.attach(model)
        print(">> AlayaJet Attached.")

    start = time.time()
    text = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=20, verbose=False)
    end = time.time()
    
    print(f"[{'AlayaJet' if use_alayajet else 'Baseline'}] Time: {end-start:.3f}s | Output: {text.strip().replace('\n', ' ')}")
    
    if engine:
        engine.detach()
        print(">> AlayaJet Detached.")
        
    return text

def main(args):
    print("=== Consistency Verification ===")
    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)
    
    prompt = "Calculate 123 + 456 and explain why."
    
    print("\n1. Running Baseline (Normal)...")
    baseline_text = run_generation(model, tokenizer, prompt, use_alayajet=False)
    
    print("\n2. Running AlayaJet (Offload)...")
    alayajet_text = run_generation(model, tokenizer, prompt, use_alayajet=True)
    
    print("\n=== Results ===")
    if baseline_text == alayajet_text:
        print("✅ SUCCESS: Outputs are identical!")
    else:
        print("❌ FAILURE: Outputs differ!")
        print(f"Base:  {baseline_text}")
        print(f"Alaya: {alayajet_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-0.5B-Instruct-4bit", help="Model path")
    args = parser.parse_args()
    
    main(args)