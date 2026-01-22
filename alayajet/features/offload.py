
import mlx.core as mx
import shutil
from pathlib import Path
from .base import AlayaFeature
from .. import patch_utils
import mlx_lm.models.cache as cache_module

class OffloadManager:
    """Manages disk storage for KV cache."""
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.reset_storage()

    def reset_storage(self):
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, layer_idx: int) -> Path:
        return self.cache_dir / f"layer_{layer_idx}.safetensors"

    def save(self, layer_idx: int, keys, values, offset):
        mx.eval(keys, values)
        data = {"keys": keys, "values": values}
        metadata = {"offset": str(offset)}
        mx.save_safetensors(str(self.get_path(layer_idx)), data, metadata)
        return None, None

    def load(self, layer_idx: int):
        path = self.get_path(layer_idx)
        if not path.exists():
            return None, None, 0
        arrays, metadata = mx.load(str(path), return_metadata=True)
        return arrays["keys"], arrays["values"], int(metadata["offset"])

class OffloadFeature(AlayaFeature):
    def __init__(self, cache_dir: str = "./kv_offload_tmp"):
        self.manager = OffloadManager(cache_dir)
        pass

    def on_attach(self, engine):
        # Apply the specific patch for KVCache.state to handle None keys
        patch_utils.patch_class_property(cache_module.KVCache, "state", self._safe_state_getter)
        print(f"[AlayaJet] OffloadFeature enabled. Dir: {self.manager.cache_dir}")

    def on_model_start(self, model):
        pass

    def on_attention_pre(self, layer_idx: int, x, mask=None, cache=None):
        if cache is None: return
        L = x.shape[1]
        
        # Decode phase (L=1): Load
        if L == 1:
            if not hasattr(cache, "keys") or cache.keys is None:
                keys, values, offset = self.manager.load(layer_idx)
                if keys is not None:
                    cache.keys = keys
                    cache.values = values
                    cache.offset = offset

    def on_attention_post(self, layer_idx: int, output, x, mask=None, cache=None):
        L = x.shape[1]
        # Prefill phase (L > 1): Save and Offload
        if L > 1:
            if cache is not None and hasattr(cache, "keys") and cache.keys is not None:
                cache.keys, cache.values = self.manager.save(
                    layer_idx, cache.keys, cache.values, cache.offset
                )
                
                if layer_idx % 4 == 0:
                     active = mx.metal.get_active_memory() / 1024**3
                     # print(f"  [Prefill L{layer_idx}] Active VRAM: {active:.2f}GB")
        
        return output

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
