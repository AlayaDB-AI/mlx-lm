import os
import mlx.core as mx
import time
import sys
import numpy as np
from ..base import AlayaFeature
from .kv_cache import QuestController
from .ops import append_kv, decode_estimate, decode_topk_np, decode_topk_frame_ids_np, decode_topk_frame_ids_from_indices_np, decode_sparse_attn, apply_rope_in_place
from .prefill_pipeline import prefill_with_kv_cache_pipelined
from .timing import QuestTiming
from ...patch_utils import replace_method, patch_class_property
import mlx_lm.models.cache as cache_module

class QuestFeature(AlayaFeature):
    def __init__(
        self,
        page_budget: int = 128,
        cache_dir: str = "./kv_quest_tmp",
        async_disk_write: bool = False,
        timing: bool = False,
        timing_sync: bool = True,
        trace_output: str | None = None,
        trace_phase: str | None = None,
        trace_decode_steps: int | None = 3,
        page_size: int = 64,
        release_active_buffer_on_prefill: bool = True,
        log_prefill_layer_timing: bool = False,
        log_prefill_io_overlap: bool = False,
        log_decode_lru_hit_rate: bool = False,
        log_memory: bool = False,
        log_memory_sync: bool = True,
        log_prefill_progress: bool = True,
        disable_os_cache: bool = True,
    ):
        super().__init__()
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.page_budget = page_budget
        self.page_size = page_size
        self.cache_dir = cache_dir
        self.async_disk_write = async_disk_write
        self.timing = QuestTiming(
            enabled=timing,
            sync=timing_sync,
            trace_output=trace_output,
            trace_phase=trace_phase,
            trace_decode_steps=trace_decode_steps,
        )
        self.release_active_buffer_on_prefill = release_active_buffer_on_prefill
        self.log_prefill_layer_timing = log_prefill_layer_timing
        self.log_prefill_io_overlap = log_prefill_io_overlap
        self.log_decode_lru_hit_rate = log_decode_lru_hit_rate
        self.log_memory = log_memory
        self.log_memory_sync = log_memory_sync
        self.log_prefill_progress = log_prefill_progress
        self.disable_os_cache = disable_os_cache
        self.controller = None
        self.max_seq_len = 32768 # Default max, can be inferred from config
        self.config = None
        self._prefill_start = None
        self._decode_step = 0
        
    def on_detach(self):
        if self.timing.enabled:
            self.timing.report(prefix="[Quest][Timing][Summary]")
        self.timing.write_trace()
        if self.controller is not None:
            self.controller.close_prefill_resources()

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
         
        def quest_forward_factory(original_forward):
            # This is the new method. `self` is the Attention layer instance.
            def new_forward(attn_self, x: mx.array, mask=None, cache=None):
                return self.forward_hook(attn_self, x, mask, cache)
            return new_forward
            
        replace_method(attn_cls, "__call__", quest_forward_factory)
        print(f"[Quest] Patched {attn_cls.__name__} with Quest Attention.")

    def _build_controller(self):
        if self.controller is not None:
            return self.controller
        config = self.config
        if config is None:
            return None

        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size

        if hasattr(config, "head_dim"):
            head_dim = config.head_dim
        elif hasattr(config, "attention_head_dim"):
            head_dim = config.attention_head_dim
        else:
            head_dim = hidden_size // num_heads

        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        self.controller = QuestController(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            page_size=self.page_size,
            page_budget=self.page_budget,
            max_seq_len=self.max_seq_len,
            num_kv_heads=num_kv_heads,
            cache_dir=self.cache_dir,
            dtype=mx.float16,
            async_disk_write=self.async_disk_write,
            disable_os_cache=self.disable_os_cache,
            release_active_buffer_on_prefill=self.release_active_buffer_on_prefill,
        )
        self.controller.prefill_io_timing = self.log_prefill_io_overlap
        self.controller._timing_hook = self.timing.record if self.timing.enabled else None
        print(
            f"[Quest] Controller Initialized: {self.page_budget} pages budget, "
            f"disk cache at {self.cache_dir}"
        )
        return self.controller

    def ensure_controller(self, cache_dir: str | None = None):
        if cache_dir and cache_dir != self.cache_dir:
            self.cache_dir = cache_dir
            if self.controller is not None:
                self.controller.close_prefill_resources()
                self.controller.kv_cache.buffer_pool.close()
                self.controller = None
        return self._build_controller()

    def on_model_start(self, model):
        if self.controller is None:
            self._build_controller()

    def _format_mem(self, value: int | None) -> str:
        if value is None:
            return "n/a"
        return f"{value / (1024 * 1024):.1f}MiB"

    def _log_mem(self, tag: str, *sync_arrays):
        if not self.log_memory:
            return
        if self.log_memory_sync and sync_arrays:
            to_sync = [arr for arr in sync_arrays if arr is not None]
            if to_sync:
                mx.eval(to_sync)
        get_active = getattr(mx, "get_active_memory", None)
        get_peak = getattr(mx, "get_peak_memory", None)
        if get_active is None or get_peak is None:
            print(f"[Quest][Mem] {tag} | stats unavailable")
            return
        active = get_active()
        peak = get_peak()
        get_cache = getattr(mx, "get_cache_memory", None)
        cache = get_cache() if get_cache is not None else None
        print(
            f"[Quest][Mem] {tag} | active={self._format_mem(active)} "
            f"peak={self._format_mem(peak)} cache={self._format_mem(cache)}"
        )

    def _update_prefill_progress(self, layer_idx: int, seq_len: int):
        if not self.log_prefill_progress:
            return
        if self.controller is None:
            return
        total_layers = self.controller.num_layers
        if total_layers <= 0:
            return
        step = min(layer_idx + 1, total_layers)
        pct = step / total_layers
        bar_len = 24
        filled = int(bar_len * pct)
        bar = "#" * filled + "-" * (bar_len - filled)
        line = (
            f"[Quest][Prefill] L={seq_len} {step}/{total_layers} "
            f"[{bar}] {pct * 100:5.1f}%"
        )
        use_inplace = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        if use_inplace:
            end = "\n" if step == total_layers else ""
            sys.stdout.write(f"\r\033[2K{line}{end}")
            sys.stdout.flush()
        else:
            print(line, flush=True)

    def forward_hook(self, attn_layer, x: mx.array, mask=None, cache=None):
        """
        The replacement forward function for Attention layers.
        """
        B, L, _ = x.shape
        phase = "prefill" if L > 1 else "decode"
        layer_t0 = time.perf_counter() if self.log_prefill_layer_timing and phase == "prefill" else None
        layer_idx = self.engine.layer_counter
        self.controller.set_prefill_write_through(phase == "prefill")
        if layer_idx == 0:
            if phase == "prefill":
                self.controller.kv_cache.buffer_pool.flush()
            if phase == "prefill":
                self._decode_step = 0
            self.timing.begin_step(phase, self._decode_step if phase == "decode" else None)
            if phase == "decode":
                self._decode_step += 1

        # 1. Projections
        if phase == "prefill":
            self._log_mem(f"{phase}_L{layer_idx:02d}_start", x)
        t_proj = time.perf_counter() if self.timing.enabled else None
        q = attn_layer.q_proj(x)
        k = attn_layer.k_proj(x)
        v = attn_layer.v_proj(x)
        
        # Reshape to (B, L, H, D)
        num_heads = attn_layer.n_heads if hasattr(attn_layer, "n_heads") else self.controller.num_heads
        # Check for GQA/MQA
        num_kv_heads = attn_layer.n_kv_heads if hasattr(attn_layer, "n_kv_heads") else num_heads
        head_dim = self.controller.head_dim
        
        q = q.reshape(B, L, num_heads, head_dim)
        k = k.reshape(B, L, num_kv_heads, head_dim)
        v = v.reshape(B, L, num_kv_heads, head_dim)

        if hasattr(attn_layer, "q_norm"):
            q = attn_layer.q_norm(q)
        if hasattr(attn_layer, "k_norm"):
            k = attn_layer.k_norm(k)
        if self.timing.enabled:
            self.timing.record(f"{phase}_proj", t_proj, q, k, v)
        if phase == "prefill":
            self._log_mem(f"{phase}_L{layer_idx:02d}_qkv", q, k, v)
        
        # 2. Quest Logic
        if layer_idx == 0:
            # Start of a model forward pass
            current_seq_len = L
            # Allocate space
            self.controller.prepare_metadata(current_seq_len)
            # Setup indices
            self.controller.begin_forward(current_seq_len)
            if phase == "prefill":
                self._prefill_start = time.perf_counter()

        # 3. RoPE
        if hasattr(attn_layer, "rope"):
            t_rope = time.perf_counter() if self.timing.enabled else None
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
            if self.timing.enabled:
                self.timing.record(f"{phase}_rope", t_rope, q, k)
            if phase == "prefill":
                self._log_mem(f"{phase}_L{layer_idx:02d}_rope", q, k)
            
        # 4. Append KV
        if B != 1:
             raise NotImplementedError("[Quest] Batch size > 1 not supported yet.")
             
        k_in = k.squeeze(0)
        v_in = v.squeeze(0)
        q_in = q.squeeze(0)
        
        # Append (No repeat needed, ops handle GQA now)
        t_append = time.perf_counter() if self.timing.enabled else None
        append_kv(k_in, v_in, self.controller, layer_idx)
        if self.timing.enabled:
            self.timing.record(f"{phase}_append_kv", t_append)
        
        # 5. Attention
        if L > 1:
            # Prefill with streamed prefix KV to support unlimited context.
            q_p = q.transpose(0, 2, 1, 3)
            k_p = k.transpose(0, 2, 1, 3)
            v_p = v.transpose(0, 2, 1, 3)

            t_prefill = time.perf_counter() if self.timing.enabled else None
            out = prefill_with_kv_cache_pipelined(
                q_p,
                k_p,
                v_p,
                controller=self.controller,
                layer_idx=layer_idx,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            if self.timing.enabled:
                self.timing.record("prefill_attn", t_prefill, out)

            # Output is (B, H, L, D) -> (B, L, H, D)
            out = out.transpose(0, 2, 1, 3)

            # Best-effort release of large temporaries.
            del q_p, k_p, v_p
            self.controller.kv_cache.offload_active_buffer_async(layer_idx)
            
        else:
            # Decode: Quest Sparse Attention
            
            # 1. Estimate
            # q_in: (1, H, D)
            need_estimate = self.controller.need_estimate()
            scores = None
            use_fused_metal_sparse = (
                os.environ.get("ALAYAJET_QUEST_METAL_SPARSE") == "1"
            )
            use_selected_page_metal = (
                os.environ.get("ALAYAJET_QUEST_METAL_SELECTED_PAGE", "0").lower()
                in ("1", "true", "yes", "on")
            )
            use_latest_resident = (
                use_selected_page_metal
                and os.environ.get("ALAYAJET_QUEST_METAL_SELECTED_FRAME", "0").lower()
                in ("resident", "resident_arena", "global")
            )
            if need_estimate and use_fused_metal_sparse and not use_selected_page_metal:
                topk = np.empty((q_in.shape[1], 0), dtype=np.int64)
            elif need_estimate:
                t_est = time.perf_counter() if self.timing.enabled else None
                scores = decode_estimate(q_in, self.controller, layer_idx)
                topk = decode_topk_np(
                    np.array(scores),
                    self.controller.inference_page_budget,
                )
                if self.timing.enabled:
                    self.timing.record("decode_estimate_topk", t_est)
            else:
                num_pages = len(self.controller.kv_indices_without_last)
                base_indices = np.arange(num_pages, dtype=np.int64)[None, :]
                topk = np.repeat(base_indices, q_in.shape[1], axis=0)

            # 2. Sparse Attn
            # q_in: (1, H, D) -> need (1, H, 1, D) for SDPA
            # Transpose (L, H, D) to (H, L, D) then expand to (1, H, L, D)
            q_sdpa = mx.expand_dims(q_in.transpose(1, 0, 2), axis=0) 
            
            t_sparse = time.perf_counter() if self.timing.enabled else None
            out_sdpa = decode_sparse_attn(
                q_sdpa,
                topk,
                self.controller,
                layer_idx,
            )
            if self.timing.enabled:
                self.timing.record("decode_sparse_attn", t_sparse, out_sdpa)
            # out_sdpa: (1, H, 1, D)
            
            # Reshape to (B, L, H, D) -> (1, 1, H, D)
            out = out_sdpa.transpose(0, 2, 1, 3) # (1, 1, H, D)
            
            # Best-effort release of large temporaries.
            del q_sdpa, out_sdpa, topk
            if scores is not None:
                del scores

        # Best-effort release of per-layer QKV intermediates.
        del q, k, v, q_in, k_in, v_in

        # 6. Output Projection
        # Reshape to (B, L, Hidden)
        t_out = time.perf_counter() if self.timing.enabled else None
        out = out.reshape(B, L, -1)
        out = attn_layer.o_proj(out)
        if self.timing.enabled:
            self.timing.record(f"{phase}_o_proj", t_out, out)
        if phase == "prefill":
            self._log_mem(f"{phase}_L{layer_idx:02d}_o_proj", out)

        if (
            self._prefill_start is not None
            and phase == "prefill"
            and layer_idx == self.controller.num_layers - 1
        ):
            self.timing.record("prefill_total", self._prefill_start)
            self._prefill_start = None

        if phase == "prefill" and self.release_active_buffer_on_prefill:
            self.controller.kv_cache.offload_active_buffer(layer_idx)
        if layer_t0 is not None:
            elapsed = (time.perf_counter() - layer_t0) * 1000.0
            print(f"[Quest][Prefill] Layer {layer_idx:02d} | L={L} | {elapsed:.2f} ms")
        if phase == "prefill":
            self._update_prefill_progress(layer_idx, L)
        if (
            phase == "decode"
            and self.log_decode_lru_hit_rate
            and layer_idx == self.controller.num_layers - 1
        ):
            hits, misses = self.controller.kv_cache.pop_lru_stats()
            selected_hits, selected_misses = self.controller.kv_cache.pop_selected_page_stats()
            total = hits + misses
            hit_rate = (hits / total * 100.0) if total else 0.0
            selected_total = selected_hits + selected_misses
            selected_hit_rate = (
                selected_hits / selected_total * 100.0 if selected_total else 0.0
            )
            step_idx = max(self._decode_step - 1, 0)
            print(
                f"[Quest][LRU] Step {step_idx} | hit {hit_rate:.2f}% "
                f"(hits={hits}, misses={misses}) | selected-page hit "
                f"{selected_hit_rate:.2f}% "
                f"(hits={selected_hits}, misses={selected_misses})"
            )
        
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
