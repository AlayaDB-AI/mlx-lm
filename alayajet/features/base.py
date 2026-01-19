
class AlayaFeature:
    """
    Base class for AlayaJet features (e.g., Offloading, Sparse Attention).
    """
    def on_attach(self, engine):
        pass

    def on_detach(self):
        pass

    def on_model_start(self, model):
        """Called at the beginning of Model.__call__"""
        pass

    def on_attention_pre(self, layer_idx: int, x, mask=None, cache=None):
        """Called before Attention computation."""
        pass

    def on_attention_post(self, layer_idx: int, output, x, mask=None, cache=None):
        """Called after Attention computation. Returns modified output."""
        return output
