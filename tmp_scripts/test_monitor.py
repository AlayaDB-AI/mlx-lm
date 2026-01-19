
import mlx.core as mx
from mlx_lm import load, generate
import sys
import os

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alayajet import AlayaEngine
from alayajet.features.monitor import VRAMMonitorFeature
from alayajet.features.offload import OffloadFeature

def main():
    print("=== AlayaJet Monitor & Offload Test ===")
    
    model_path = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    
    # Create Engine
    engine = AlayaEngine()
    
    # Add Features manually to test composition
    print("Adding OffloadFeature...")
    engine.add_feature(OffloadFeature(cache_dir="./kv_monitor_test"))
    
    print("Adding VRAMMonitorFeature...")
    engine.add_feature(VRAMMonitorFeature(interval=2)) # Print every 2 layers
    
    engine.attach(model)
    
    prompt = "Describe the view from the top of a mountain."
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        add_generation_prompt=True
    )
    
    print("\nStarting generation...")
    # Generate a few tokens
    generate(model, tokenizer, prompt=formatted_prompt, max_tokens=5, verbose=True)
    
    engine.detach()

if __name__ == "__main__":
    main()
