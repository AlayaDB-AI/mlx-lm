from . import patch_utils
from typing import List
from .features.base import AlayaFeature
from .features.offload import OffloadFeature
from .features.monitor import VRAMMonitorFeature
from .features.chunking import ChunkComputationFeature
from .features.quest.integration import QuestFeature

class AlayaEngine:
    def __init__(self):
        self.model = None
        self.features: List[AlayaFeature] = []
        self.layer_counter = 0

    def add_feature(self, feature: AlayaFeature):
        self.features.append(feature)

    def attach(self, model):
        if self.model is not None:
            self.detach()
        self.model = model

        # Notify features
        for f in self.features:
            f.on_attach(self)

        # Patch Model Hooks
        patch_utils.patch_model(
            model,
            attention_pre_hook=self._dispatch_attention_pre,
            attention_post_hook=self._dispatch_attention_post,
            model_pre_hook=self._dispatch_model_pre
        )
        print("[AlayaJet] Engine attached with features:", [type(f).__name__ for f in self.features])

    def detach(self):
        if self.model:
            for f in self.features:
                f.on_detach()
            patch_utils.restore_all()
            self.model = None
            print("[AlayaJet] Engine detached.")

    # --- Dispatchers ---

    def _dispatch_model_pre(self, _self, *args, **kwargs):
        self.layer_counter = 0
        for f in self.features:
            f.on_model_start(_self)

    def _dispatch_attention_pre(self, _self, x, mask=None, cache=None):
        # We pass the current layer counter to features
        for f in self.features:
            f.on_attention_pre(self.layer_counter, x, mask, cache)

    def _dispatch_attention_post(self, _self, output, x, mask=None, cache=None):
        for f in self.features:
            output = f.on_attention_post(self.layer_counter, output, x, mask, cache)
        
        # Increment global layer counter
        self.layer_counter += 1
        return output

    # --- Convenience Methods ---
    
    @classmethod
    def with_offload(cls, cache_dir: str = "./kv_offload_tmp"):
        engine = cls()
        engine.add_feature(OffloadFeature(cache_dir))
        return engine

    @classmethod
    def with_monitor(cls, interval: int = 4):
        engine = cls()
        engine.add_feature(VRAMMonitorFeature(interval=interval))
        return engine

    @classmethod
    def with_chunking(cls, chunk_size: int = 4096):
        engine = cls()
        engine.add_feature(ChunkComputationFeature(chunk_size=chunk_size))
        return engine

    @classmethod
    def with_quest(
        cls,
        page_budget: int = 128,
        cache_dir: str = "./kv_quest_tmp",
        async_disk_write: bool = False,
        timing: bool = False,
        timing_sync: bool = True,
        trace_output: str | None = None
    ):
        engine = cls()
        engine.add_feature(
            QuestFeature(
                page_budget=page_budget,
                cache_dir=cache_dir,
                async_disk_write=async_disk_write,
                timing=timing,
                timing_sync=timing_sync,
                trace_output=trace_output
            )
        )
        return engine
