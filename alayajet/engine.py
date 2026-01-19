
import mlx_lm.models.cache as cache_module
from . import patch_utils
from .offload import OffloadManager

class AlayaEngine:
    """
    The main controller for AlayaJet features (Offloading, Sparse Attention, etc.).
    It manages the lifecycle of hooks and patches attached to a model.
    """
    def __init__(self, cache_dir: str = "./kv_offload_tmp"):
        self.offload_manager = OffloadManager(cache_dir)
        self.model = None

    def attach(self, model):
        """
        Enable AlayaJet features for the given model.
        """
        if self.model is not None:
            print("[AlayaJet] Warning: Engine already attached to a model. Detaching first.")
            self.detach()

        self.model = model
        
        # Patch Model & Attention with instance methods as hooks
        patch_utils.patch_model(
            model,
            attention_pre_hook=self._attention_pre_hook,
            attention_post_hook=self._attention_post_hook,
            model_pre_hook=self._model_pre_hook
        )
        
        # Patch KVCache state property globally (for now)
        # TODO: Ideally this should be scoped, but class property patching is global.
        patch_utils.patch_class_property(cache_module.KVCache, "state", self._safe_state_getter)
        
        print(f"[AlayaJet] Engine attached. Offloading to {self.offload_manager.cache_dir}")

    def detach(self):
        """
        Disable AlayaJet features and restore original model behavior.
        """
        if self.model:
            patch_utils.restore_all()
            self.model = None
            # self.offload_manager.reset_storage() # Optional: keep files for inspection
            print("[AlayaJet] Engine detached.")

    # --- Hooks ---

    def _model_pre_hook(self, _self, *args, **kwargs):
        """Hook for Model.__call__ (start of generation step)."""
        self.offload_manager.reset_counter()

    def _attention_pre_hook(self, _self, x, mask=None, cache=None):
        """Hook for Attention.__call__ (before computation)."""
        if cache is None:
            return
        
        L = x.shape[1]
        
        # Decode phase (L=1): Load if currently offloaded
        if L == 1:
            if not hasattr(cache, "keys") or cache.keys is None:
                keys, values, offset = self.offload_manager.load_layer(self.offload_manager.layer_counter)
                if keys is not None:
                    cache.keys = keys
                    cache.values = values
                    cache.offset = offset

    def _attention_post_hook(self, _self, output, x, mask=None, cache=None):
        """Hook for Attention.__call__ (after computation)."""
        L = x.shape[1]
        
        # Prefill phase (L > 1): Save and Offload (Clear Memory)
        if L > 1:
            if cache is not None and hasattr(cache, "keys") and cache.keys is not None:
                cache.keys, cache.values = self.offload_manager.save_layer(
                    self.offload_manager.layer_counter, cache.keys, cache.values, cache.offset
                )

        # Increment layer counter for the next layer in the stack
        self.offload_manager.layer_counter += 1
        return output

    # --- Helpers ---

    @staticmethod
    def _safe_state_getter(self_cache):
        """
        Replacement for KVCache.state to handle offloaded (None) keys.
        'self_cache' is the KVCache instance.
        """
        if self_cache.keys is None:
            return []
        
        if self_cache.offset == self_cache.keys.shape[2]:
            return self_cache.keys, self_cache.values
        else:
            return (
                self_cache.keys[..., : self_cache.offset, :],
                self_cache.values[..., : self_cache.offset, :],
            )
