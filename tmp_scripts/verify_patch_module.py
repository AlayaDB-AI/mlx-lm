import mlx.core as mx
from mlx_lm import load, generate
import sys
import os

# Ensure root directory is in python path to import alayajet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alayajet import patch_utils

# --- 定义 Hooks ---

def attention_pre_hook(self, x, mask=None, cache=None):
    # 注意：新的 patch_utils 传递 *args, **kwargs
    # 这里我们根据参数位置或关键字来获取
    
    # x 通常是第一个位置参数 args[0]
    # mask, cache 可能是位置参数或关键字参数
    
    # 简单起见，我们只打印日志，不尝试修改参数
    if not hasattr(self, "_patch_logged"):
        print(f"[Hook] Attention Input shape: {x.shape} | L={x.shape[1]}")
        self._patch_logged = True
    return None 

def model_pre_hook(self, *args, **kwargs):
    # 只有当这是顶层调用时我们才打印（避免递归调用打印太多）
    print("\n[Hook] Model.__call__ started")

# --- 主程序 ---

def main():
    model_path = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    print(f"1. Loading model: {model_path}")
    model, tokenizer = load(model_path)
    
    print("2. Applying dynamic patches to the loaded model instance...")
    patch_utils.patch_model(
        model, 
        attention_pre_hook=attention_pre_hook,
        model_pre_hook=model_pre_hook
    )
    
    prompt = "Hello!"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    print("3. Generating...")
    generate(model, tokenizer, prompt=formatted_prompt, max_tokens=5, verbose=True)
    
    print("\n4. Restoring patches...")
    patch_utils.restore_all()

if __name__ == "__main__":
    main()