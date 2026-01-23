import time
from collections import defaultdict

import mlx.core as mx

from .base import AlayaFeature
from .. import patch_utils


class TimingFeature(AlayaFeature):
    def __init__(self, timing_sync: bool = True, track_mlp: bool = True, track_norm: bool = False):
        self.timing_sync = timing_sync
        self.track_mlp = track_mlp
        self.track_norm = track_norm
        self._stats = defaultdict(lambda: [0.0, 0])
        self._attn_start = {}

    def _record(self, key: str, start_time: float, *sync_arrays):
        if self.timing_sync and sync_arrays:
            to_sync = [arr for arr in sync_arrays if arr is not None]
            if to_sync:
                mx.eval(to_sync)
        elapsed = time.perf_counter() - start_time
        stat = self._stats[key]
        stat[0] += elapsed
        stat[1] += 1

    def _make_timed_call(self, key: str):
        def factory(original_call):
            def timed_call(layer_self, *args, **kwargs):
                t0 = time.perf_counter()
                out = original_call(layer_self, *args, **kwargs)
                self._record(key, t0, out)
                return out
            return timed_call
        return factory

    def on_attach(self, engine):
        self.engine = engine
        model = engine.model
        layers = getattr(model, "layers", None)
        if layers is None and hasattr(model, "model"):
            layers = getattr(model.model, "layers", None)
        if not layers:
            return

        first_layer = layers[0]

        if self.track_mlp and hasattr(first_layer, "mlp"):
            mlp_cls = type(first_layer.mlp)
            patch_utils.replace_method(mlp_cls, "__call__", self._make_timed_call("mlp"))

        if self.track_norm:
            for attr in ("input_layernorm", "post_attention_layernorm"):
                if hasattr(first_layer, attr):
                    norm_cls = type(getattr(first_layer, attr))
                    patch_utils.replace_method(norm_cls, "__call__", self._make_timed_call("norm"))

    def on_attention_pre(self, layer_idx: int, x, mask=None, cache=None):
        is_prefill = x.shape[1] > 1
        self._attn_start[layer_idx] = (time.perf_counter(), is_prefill)

    def on_attention_post(self, layer_idx: int, output, x, mask=None, cache=None):
        start = self._attn_start.pop(layer_idx, None)
        if start is None:
            return output
        t0, is_prefill = start
        key = "attn_prefill" if is_prefill else "attn_decode"
        self._record(key, t0, output)
        return output

    def report(self, reset: bool = False, prefix: str = "[Timing]"):
        if not self._stats:
            return
        print(prefix)
        for key in sorted(self._stats.keys()):
            total, count = self._stats[key]
            avg = total / count if count else 0.0
            print(f"  {key}: total {total:.3f}s, avg {avg*1000:.3f}ms, n={count}")
        if reset:
            self._stats.clear()

    def on_detach(self):
        self.report(prefix="[Timing][Summary]")
