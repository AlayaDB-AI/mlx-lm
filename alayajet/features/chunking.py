
import mlx.core as mx
from .base import AlayaFeature
from .. import patch_utils

class ChunkComputationFeature(AlayaFeature):
    """
    Splits MLP and LayerNorm computations into chunks to reduce peak memory during prefill.
    """
    def __init__(self, chunk_size: int = 4096, verbose: bool = True):
        self.chunk_size = chunk_size
        self.verbose = verbose

    def on_attach(self, engine):
        model = engine.model
        if not model:
            return

        layers = getattr(model, "layers", None)
        if layers is None and hasattr(model, "model"):
            layers = getattr(model.model, "layers", None)
            
        if not layers or len(layers) == 0:
            return

        first_layer = layers[0]
        
        # 1. Patch MLP
        if hasattr(first_layer, "mlp"):
            mlp_cls = type(first_layer.mlp)
            patch_utils.replace_method(mlp_cls, "__call__", self._make_chunked_call)
            if self.verbose:
                print(f"[AlayaJet] Chunking enabled for {mlp_cls.__name__} (Size: {self.chunk_size})")

        # 2. Patch Norm
        norm_attr = None
        if hasattr(first_layer, "input_layernorm"):
            norm_attr = "input_layernorm"
        elif hasattr(first_layer, "post_attention_layernorm"):
            norm_attr = "post_attention_layernorm"
            
        if norm_attr:
            norm_cls = type(getattr(first_layer, norm_attr))
            patch_utils.replace_method(norm_cls, "__call__", self._make_chunked_call)
            if self.verbose:
                print(f"[AlayaJet] Chunking enabled for {norm_cls.__name__} (Size: {self.chunk_size})")

    def _make_chunked_call(self, original_call):
        def chunked_call(layer_self, x, *args, **kwargs):
            if x.ndim != 3:
                return original_call(layer_self, x, *args, **kwargs)
            
            L = x.shape[1]
            if L <= self.chunk_size:
                return original_call(layer_self, x, *args, **kwargs)
            
            results = []
            for i in range(0, L, self.chunk_size):
                chunk_x = x[:, i : i + self.chunk_size, :]
                
                # Compute
                res = original_call(layer_self, chunk_x, *args, **kwargs)
                
                # Critical: Force computation to free intermediate activation graphs
                mx.eval(res)
                
                results.append(res)
            
            # Combine results
            return mx.concatenate(results, axis=1)
            
        return chunked_call
