import mlx.core as mx
import mlx.nn as nn
from alayajet.engine import AlayaEngine

# Define a Dummy Config and Model that mimics Llama structure
class DummyConfig:
    def __init__(self):
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.hidden_size = 32
        self.max_position_embeddings = 128

class DummyAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.n_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
    def __call__(self, x, mask=None, cache=None):
        # Standard Attention Logic (Simplified)
        print("Running Original Attention...")
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        B, L, D = q.shape
        out = mx.random.normal((B, L, D))
        return self.o_proj(out)

class DummyLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = DummyAttention(config)

class DummyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = [DummyLayer(config) for _ in range(config.num_hidden_layers)]
        
    def __call__(self, x):
        # Simple forward pass
        for layer in self.layers:
            x = layer.self_attn(x)
        return x

def main():
    print("--- Setting up Quest Reproduction ---")
    config = DummyConfig()
    model = DummyModel(config)
    
    # Create Engine with Quest
    engine = AlayaEngine.with_quest(page_budget=2, cache_dir="./test_quest_integration")
    
    print("\n--- Attaching Engine ---")
    engine.attach(model)
    
    # Simulate Generation Step 1 (Prefill)
    print("\n--- Step 1: Prefill (L=10) ---")
    x = mx.random.normal((1, 10, 32)) # B=1, L=10, D=32
    output = model(x)
    print("Output Shape:", output.shape)
    
    # Simulate Generation Step 2 (Decode)
    print("\n--- Step 2: Decode (L=1) ---")
    x = mx.random.normal((1, 1, 32))
    output = model(x)
    print("Output Shape:", output.shape)
    
    # Simulate Generation Step 3 (Decode)
    print("\n--- Step 3: Decode (L=1) ---")
    output = model(x)
    print("Output Shape:", output.shape)

    print("\n--- Detaching Engine ---")
    engine.detach()

    # Verify Original Logic Restored
    print("\n--- Verification: Original Logic ---")
    output = model(x) # Should print "Running Original Attention..."

if __name__ == "__main__":
    main()
