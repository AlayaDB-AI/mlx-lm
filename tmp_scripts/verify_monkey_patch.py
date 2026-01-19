
import mlx.core as mx
from mlx_lm import load, generate
import mlx_lm.models.qwen2 as qwen2_model

# 1. 保存原始的 Attention.__call__ 方法
original_attention_call = qwen2_model.Attention.__call__

# 2. 定义新的带有日志打印的方法
def monkey_patched_attention_call(self, x, mask=None, cache=None):
    # 打印一条日志证明补丁生效 (为了避免刷屏，只在第一次调用或特定条件下打印)
    # 这里为了演示，我们简单地打印一下当前层的相关信息
    # 注意：self 是 Attention 实例
    
    # 我们可以尝试推断这是第几层，或者简单打印 "Patch Active"
    if not hasattr(self, "_logged_patch"):
        print(f"[MonkeyPatch] Attention called for one of the layers. Input shape: {x.shape}")
        self._logged_patch = True
    
    # 3. 调用原始方法，保持原有逻辑不变
    return original_attention_call(self, x, mask, cache)

# 4. 应用猴子补丁
print("Applying Monkey Patch to mlx_lm.models.qwen2.Attention...")
qwen2_model.Attention.__call__ = monkey_patched_attention_call

def main():
    model_path = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    print(f"Loading model: {model_path}")
    
    # 加载模型
    model, tokenizer = load(model_path)
    
    # 构造输入
    prompt = "Hello, who are you?"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    print("Starting generation...")
    # 生成文本，这将触发 Attention 调用
    response = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=10, verbose=True)
    print("\nGeneration finished.")

if __name__ == "__main__":
    main()
