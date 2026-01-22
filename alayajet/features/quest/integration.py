import mlx.core as mx
import mlx.nn as nn
from ..base import AlayaFeature
from .kv_cache import QuestController
from .ops import append_kv, decode_estimate, decode_topk, decode_sparse_attn, apply_rope_in_place
from ...patch_utils import replace_method, patch_class_property
import mlx_lm.models.cache as cache_module

class QuestFeature(AlayaFeature):
    def __init__(self, page_budget: int = 128, cache_dir: str = "./kv_quest_tmp"):
        super().__init__()
        self.page_budget = page_budget
        self.cache_dir = cache_dir
        self.controller = None
        self.max_seq_len = 32768 # Default max, can be inferred from config
        self.config = None
        
    def on_attach(self, engine):
        self.engine = engine
        self.model = engine.model
        
        # Patch KVCache.state to handle None keys safely (fix for mlx_lm.generate)
        patch_class_property(cache_module.KVCache, "state", self._safe_state_getter)
        
        # Infer config
        # mlx_lm models usually store configuration in 'args' or 'config'
        self.config = getattr(self.model, "config", None)
        if self.config is None:
            self.config = getattr(self.model, "args", None)
            
        if self.config:
            # Check for max position embeddings
            if hasattr(self.config, "max_position_embeddings"):
                self.max_seq_len = self.config.max_position_embeddings
            elif hasattr(self.config, "max_sequence_length"): # Qwen/others might use this
                self.max_seq_len = self.config.max_sequence_length
            
        # We need to replace the Attention class's __call__ method completely
        # because Quest changes the internal logic significantly (no standard cache usage).
        # We use replace_method from patch_utils
        
        # Identify Attention Class
        # Similar to patch_model logic
        layers = getattr(self.model, "layers", []) or getattr(self.model.model, "layers", [])
        if not layers:
            print("[Quest] Could not find layers to patch.")
            return
            
        first_layer = layers[0]
        if not hasattr(first_layer, "self_attn"):
             print("[Quest] Could not find self_attn in layers.")
             return
             
        attn_cls = type(first_layer.self_attn)
        
        # Replace __call__ with our quest_attention_forward
        # We need a factory that takes original method (we might ignore it or use parts of it)
        # But wait, original method has `self`, we need to capture `self` (the layer instance).
        
        def quest_forward_factory(original_forward):
            # This is the new method. `self` is the Attention layer instance.
            def new_forward(attn_self, x: mx.array, mask=None, cache=None):
                return self.forward_hook(attn_self, x, mask, cache)
            return new_forward
            
        replace_method(attn_cls, "__call__", quest_forward_factory)
        print(f"[Quest] Patched {attn_cls.__name__} with Quest Attention.")

    def on_model_start(self, model):
        # Initialize Controller if not ready or dimensions changed?
        # We need model dimensions.
        if self.controller is None:
            # Get dimensions from config or first layer
            # We assume homogeneous layers
            config = self.config
            
            num_layers = config.num_hidden_layers
            num_heads = config.num_attention_heads
            hidden_size = config.hidden_size
            
            # Determine head_dim: prefer config value, fallback to calculation
            if hasattr(config, "head_dim"):
                head_dim = config.head_dim
            elif hasattr(config, "attention_head_dim"): # DeepSeek/Others
                head_dim = config.attention_head_dim
            else:
                head_dim = hidden_size // num_heads
            
            # Identify KV heads
            num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
            
            # Page size - Quest default 64?
            page_size = 64
            
            self.controller = QuestController(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                page_size=page_size,
                page_budget=self.page_budget,
                max_seq_len=self.max_seq_len,
                num_kv_heads=num_kv_heads,
                cache_dir=self.cache_dir,
                dtype=mx.float16 # TODO: Match model dtype
            )
            print(f"[Quest] Controller Initialized: {self.page_budget} pages budget, disk cache at {self.cache_dir}")
        
        pass

    def forward_hook(self, attn_layer, x: mx.array, mask=None, cache=None):
        """
        The replacement forward function for Attention layers.
        """
        # 1. Projections
        q = attn_layer.q_proj(x)
        k = attn_layer.k_proj(x)
        v = attn_layer.v_proj(x)
        
        # Reshape to (B, L, H, D)
        B, L, _ = q.shape
        num_heads = attn_layer.n_heads if hasattr(attn_layer, "n_heads") else self.controller.num_heads
        # Check for GQA/MQA
        num_kv_heads = attn_layer.n_kv_heads if hasattr(attn_layer, "n_kv_heads") else num_heads
        head_dim = self.controller.head_dim
        
        q = q.reshape(B, L, num_heads, head_dim)
        k = k.reshape(B, L, num_kv_heads, head_dim)
        v = v.reshape(B, L, num_kv_heads, head_dim)
        
        # 2. Quest Logic
        layer_idx = self.engine.layer_counter
        
        if layer_idx == 0:
            # Start of a model forward pass
            current_seq_len = L
            # Allocate space
            self.controller.prepare_metadata(current_seq_len)
            # Setup indices
            self.controller.begin_forward(current_seq_len)

        # 3. RoPE
        if hasattr(attn_layer, "rope"):
            # prepare_metadata increments seq_len by L (for the current batch)
            # So the correct starting offset for this batch is seq_len - L
            offset = self.controller.kv_cache.seq_len - L
            if hasattr(attn_layer.rope, "__call__"):
                 q = q.transpose(0, 2, 1, 3)
                 k = k.transpose(0, 2, 1, 3)
                 q = attn_layer.rope(q, offset=offset)
                 k = attn_layer.rope(k, offset=offset)
                 q = q.transpose(0, 2, 1, 3)
                 k = k.transpose(0, 2, 1, 3)
            
        # 4. Append KV
        if B != 1:
             raise NotImplementedError("[Quest] Batch size > 1 not supported yet.")
             
        k_in = k.squeeze(0)
        v_in = v.squeeze(0)
        q_in = q.squeeze(0)
        
        # Append (No repeat needed, ops handle GQA now)
        append_kv(k_in, v_in, self.controller, layer_idx)
        
        # 5. Attention
        if L > 1:
            # Simplified Prefill: Just compute attention on current chunk (q, k, v)
            # Using MLX scaled_dot_product_attention on current inputs
            # Transpose to (B, H, L, D)
            q_p = q.transpose(0, 2, 1, 3)
            k_p = k.transpose(0, 2, 1, 3)
            v_p = v.transpose(0, 2, 1, 3)
            
            # Handle GQA for Prefill: Explicitly repeat KV heads
            # MLX SDPA might not support implicit GQA broadcasting for H_kv > 1
            if num_kv_heads != num_heads:
                n_rep = num_heads // num_kv_heads
                k_p = mx.repeat(k_p, n_rep, axis=1)
                v_p = mx.repeat(v_p, n_rep, axis=1)
            
            # Causal mask
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
            mask = mask.astype(q_p.dtype)
            
            out = mx.fast.scaled_dot_product_attention(q_p, k_p, v_p, scale=1.0/mx.sqrt(head_dim), mask=mask)
            
            # Output is (B, H, L, D) -> (B, L, H, D)
            out = out.transpose(0, 2, 1, 3)
            
        else:
            # Decode: Quest Sparse Attention
            
            # 1. Estimate
            # q_in: (1, H, D)
            if self.controller.need_estimate():
                scores = decode_estimate(q_in, self.controller, layer_idx)
                topk = decode_topk(scores, self.controller.inference_page_budget)
            else:
                scores = decode_estimate(q_in, self.controller, layer_idx)
                topk = decode_topk(scores, self.controller.inference_page_budget)

            # 2. Sparse Attn
            # q_in: (1, H, D) -> need (1, H, 1, D) for SDPA
            # Transpose (L, H, D) to (H, L, D) then expand to (1, H, L, D)
            q_sdpa = mx.expand_dims(q_in.transpose(1, 0, 2), axis=0) 
            
            out_sdpa = decode_sparse_attn(q_sdpa, topk, self.controller, layer_idx)
            # out_sdpa: (1, H, 1, D)
            
            # Reshape to (B, L, H, D) -> (1, 1, H, D)
            out = out_sdpa.transpose(0, 2, 1, 3) # (1, 1, H, D)

        # 6. Output Projection
        # Reshape to (B, L, Hidden)
        out = out.reshape(B, L, -1)
        out = attn_layer.o_proj(out)
        
        # Manually increment layer counter since we bypassed the engine hook
        self.engine.layer_counter += 1
        
        return out

    @staticmethod
    def _safe_state_getter(self_cache):
        if self_cache.keys is None: return []
        if self_cache.offset == self_cache.keys.shape[2]:
            return self_cache.keys, self_cache.values
        else:
            return (
                self_cache.keys[..., : self_cache.offset, :],
                self_cache.values[..., : self_cache.offset, :],
            )
