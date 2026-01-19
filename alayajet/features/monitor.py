
import mlx.core as mx
from .base import AlayaFeature

class VRAMMonitorFeature(AlayaFeature):
    def __init__(self, interval: int = 4, verbose: bool = True):
        self.interval = interval
        self.verbose = verbose

    def on_model_start(self, model):
        if self.verbose:
            print("[Monitor] --- Model Forward Start ---")

    def on_attention_post(self, layer_idx: int, output, x, mask=None, cache=None):
        if layer_idx % self.interval == 0:
            # FORCE EVAL to get accurate memory reading at this point
            mx.eval(output)
            
            active_gb = mx.get_active_memory() / 1024**3
            peak_gb = mx.get_peak_memory() / 1024**3
            L = x.shape[1]
            phase = "Prefill" if L > 1 else "Decode"
            
            if self.verbose:
                print(f"[Monitor] {phase:7s} (L={L:5d}) | Layer {layer_idx:02d} | Active: {active_gb:5.2f} GB | Peak: {peak_gb:5.2f} GB")
        
        return output
