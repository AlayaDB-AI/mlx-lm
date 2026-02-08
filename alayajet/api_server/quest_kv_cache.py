import json
import os
import threading
from typing import List, Optional, Tuple

import numpy as np

from ..features.quest.kv_cache import QuestController


class QuestDiskCache:
    def __init__(self):
        self._locks = {}
        self._locks_lock = threading.Lock()

    def _get_lock(self, path: str) -> threading.Lock:
        with self._locks_lock:
            lock = self._locks.get(path)
            if lock is None:
                lock = threading.Lock()
                self._locks[path] = lock
            return lock

    @staticmethod
    def _resolve_paths(path: str) -> Tuple[str, str, str]:
        if path.endswith(".json"):
            cache_dir = os.path.dirname(path) or "."
            state_path = path
        else:
            cache_dir = path
            state_path = os.path.join(cache_dir, "quest_state.json")
        metadata_path = os.path.join(cache_dir, "quest_metadata.npy")
        return cache_dir, state_path, metadata_path

    @staticmethod
    def resolve_cache_dir(path: str) -> str:
        cache_dir, _, _ = QuestDiskCache._resolve_paths(path)
        return cache_dir

    @staticmethod
    def _normalize_tokens(tokens: List[int]) -> List[int]:
        return [int(t) for t in tokens]

    @staticmethod
    def _page_count(seq_len: int, page_size: int) -> int:
        if seq_len <= 0:
            return 0
        return (seq_len + page_size - 1) // page_size

    def _check_compat(self, controller: QuestController, state: dict) -> bool:
        return (
            state.get("page_size") == controller.page_size
            and state.get("num_layers") == controller.num_layers
            and state.get("num_kv_heads") == controller.num_kv_heads
            and state.get("head_dim") == controller.head_dim
            and state.get("capacity") == controller.kv_cache.capacity
            and state.get("dtype") == np.dtype(controller.kv_cache.dtype).name
        )

    def load(
        self,
        controller: QuestController,
        path: str,
        prompt_tokens: List[int],
        model_id: Optional[str] = None,
    ) -> Tuple[Optional[List[int]], List[int], int]:
        cache_dir, state_path, metadata_path = self._resolve_paths(path)
        if not os.path.exists(state_path):
            return None, prompt_tokens, 0

        lock = self._get_lock(state_path)
        with lock:
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except (json.JSONDecodeError, OSError):
                return None, prompt_tokens, 0

            if model_id and state.get("model") and state.get("model") != model_id:
                return None, prompt_tokens, 0

            if not self._check_compat(controller, state):
                return None, prompt_tokens, 0

            cached_tokens = state.get("prompt_tokens")
            if not isinstance(cached_tokens, list):
                return None, prompt_tokens, 0
            cached_tokens = self._normalize_tokens(cached_tokens)
            prompt_tokens = self._normalize_tokens(prompt_tokens)

            if not os.path.exists(metadata_path):
                return None, prompt_tokens, 0

            try:
                metadata = np.load(metadata_path, allow_pickle=False)
            except OSError:
                return None, prompt_tokens, 0

        cached_len = len(cached_tokens)
        prompt_len = len(prompt_tokens)

        max_len = min(prompt_len, cached_len)
        lcp_len = 0
        while lcp_len < max_len:
            if prompt_tokens[lcp_len] != cached_tokens[lcp_len]:
                break
            lcp_len += 1
        if lcp_len == 0:
            return None, prompt_tokens, 0
        prefix_len = lcp_len
        rest = prompt_tokens[lcp_len:]

        # If there's no suffix, keep one token to drive generation and
        # rewind the cache by one token so logits align with the prompt end.
        if not rest and prefix_len > 0:
            prefix_len -= 1
            rest = [prompt_tokens[prefix_len]]

        active_indices = state.get("active_indices")
        if not isinstance(active_indices, list):
            return None, prompt_tokens, 0
        active_indices = [int(x) for x in active_indices]

        needed_pages = self._page_count(prefix_len, controller.page_size)
        if needed_pages > len(active_indices):
            return None, prompt_tokens, 0

        if metadata.ndim != 5:
            return None, prompt_tokens, 0
        if metadata.shape[0] != controller.num_layers:
            return None, prompt_tokens, 0
        if metadata.shape[2] != 2:
            return None, prompt_tokens, 0
        if metadata.shape[3] != controller.num_kv_heads:
            return None, prompt_tokens, 0
        if metadata.shape[4] != controller.head_dim:
            return None, prompt_tokens, 0

        if metadata.shape[1] < needed_pages:
            return None, prompt_tokens, 0

        active_indices = active_indices[:needed_pages]
        metadata = metadata[:, :needed_pages]

        controller.clean_states()
        kv_cache = controller.kv_cache
        kv_cache.seq_len = prefix_len
        kv_cache.active_indices = list(active_indices)
        kv_cache.free_slots = set(range(kv_cache.capacity)) - set(active_indices)
        kv_cache.active_buffers_mx.clear()
        kv_cache.page_cached_all.fill(False)
        kv_cache._last_page_written_layers = (
            set(range(controller.num_layers)) if prefix_len > 0 else set()
        )

        controller.metadata_pool[:, active_indices] = metadata

        return cached_tokens, rest, prefix_len

    def save(
        self,
        controller: QuestController,
        path: str,
        prompt_tokens: List[int],
        model_id: Optional[str] = None,
    ) -> None:
        cache_dir, state_path, metadata_path = self._resolve_paths(path)
        os.makedirs(cache_dir, exist_ok=True)

        prompt_tokens = self._normalize_tokens(prompt_tokens)
        prompt_len = len(prompt_tokens)
        page_count = self._page_count(prompt_len, controller.page_size)

        active_indices = controller.kv_cache.active_indices[:page_count]
        if page_count and len(active_indices) < page_count:
            page_count = len(active_indices)
            prompt_len = min(prompt_len, page_count * controller.page_size)

        kv_cache = controller.kv_cache
        if kv_cache.active_indices:
            kv_cache.flush_active_buffers(kv_cache.active_indices[-1])
        kv_cache.buffer_pool.flush()

        if page_count > 0:
            metadata = controller.metadata_pool[:, active_indices]
        else:
            metadata = np.empty(
                (controller.num_layers, 0, 2, controller.num_kv_heads, controller.head_dim),
                dtype=np.float32,
            )

        state = {
            "version": 1,
            "model": model_id,
            "seq_len": prompt_len,
            "prompt_tokens": prompt_tokens,
            "page_size": controller.page_size,
            "num_layers": controller.num_layers,
            "num_kv_heads": controller.num_kv_heads,
            "head_dim": controller.head_dim,
            "capacity": controller.kv_cache.capacity,
            "dtype": np.dtype(controller.kv_cache.dtype).name,
            "active_indices": [int(x) for x in active_indices],
        }

        lock = self._get_lock(state_path)
        with lock:
            np.save(metadata_path, metadata)
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f)
