import mlx.core as mx
from mlx_lm import load, generate
from alayajet.engine import AlayaEngine
import os
import shutil
import random

def generate_passkey_prompt(length=2000):
    """
    Generates a "needle in a haystack" prompt.
    """
    # Random passkey
    passkey = str(random.randint(10000, 99999))
    
    # Filler text (repeated)
    filler = "The grass is green. The sky is blue. The sun is yellow. " * 50
    
    # Construct prompt
    # 1. Intro
    prompt = "Below is a long text containing a secret passkey. Find the passkey and ignore the filler text.\n\n"
    
    # 2. Haystack (Pre-key)
    current_len = len(prompt.split())
    while current_len < length // 2:
        prompt += filler
        current_len += len(filler.split())
        
    # 3. Needle
    needle = f"\nThe secret passkey is {passkey}.\n"
    prompt += needle
    
    # 4. Haystack (Post-key)
    while current_len < length:
        prompt += filler
        current_len += len(filler.split())
        
    # 5. Question
    prompt += "\n\nWhat is the secret passkey? The passkey is:"
    
    return prompt, passkey

def main():
    model_id = "mlx-community/Qwen2.5-7B-Instruct-1M-4bit"
    # model_id = "mlx-community/Qwen3-0.6B-bf16"
    print(f"--- Loading Model: {model_id} ---")
    
    # 1. Load Model
    model, tokenizer = load(model_id)
    
    # 2. Setup Engine with Quest
    # Use a budget that is significantly smaller than the prompt length (e.g. 10%)
    # If prompt is ~2000 tokens / 64 = 32 pages.
    # Budget 4 means 12.5% retention.
    page_budget = 4
    cache_dir = "./qwen_quest_cache"
    
    # Clean up previous cache if exists
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        
    print(f"--- Attaching Quest Engine (Budget: {page_budget} pages) ---")
    engine = AlayaEngine.with_quest(page_budget=page_budget, cache_dir=cache_dir, async_disk_write=True)
    engine.attach(model)
    
    # 3. Generate
    prompt_text, expected_passkey = generate_passkey_prompt(length=3000) # Long enough to trigger Quest
    
    messages = [{"role": "user", "content": prompt_text}]
    
    # Qwen chat template handling
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        prompt_formatted = prompt_text 
    
    print(f"\n--- Starting Generation (Length ~{len(prompt_formatted)//4} tokens) ---")
    print(f"Expected Answer: {expected_passkey}")
    
    # Using mlx_lm.generate
    # Large prefill_step_size to force single-batch prefill (simplifies logic)
    response = generate(model, tokenizer, prompt=prompt_formatted, max_tokens=50, verbose=True, prefill_step_size=1000000)
    
    print("\n--- Generation Complete ---")
    print(f"Model Output: {response}")
    
    if expected_passkey in response:
        print("\n✅ SUCCESS: Passkey retrieved!")
    else:
        print("\n❌ FAILURE: Passkey not found.")
    
    # 4. Detach
    engine.detach()
    
    # Clean up
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

if __name__ == "__main__":
    main()