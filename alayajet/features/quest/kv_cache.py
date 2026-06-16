import mlx.core as mx
import numpy as np
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict, Set, Callable

from .native import (
    resident_arena_write_batched,
    resident_write as native_resident_write,
    resident_write_batched as native_resident_write_batched,
)

try:
    import fcntl
except Exception:  # pragma: no cover - optional on some platforms
    fcntl = None


_NUMPY_TO_MX_DTYPE = {
    np.dtype(np.float16): mx.float16,
    np.dtype(np.float32): mx.float32,
    np.dtype(np.int32): mx.int32,
    np.dtype(np.int64): mx.int64,
    np.dtype(np.bool_): mx.bool_,
}


def safe_to_numpy(value, dtype=None, copy: bool = False) -> np.ndarray:
    """Convert MLX tensors to NumPy without relying on fragile buffer exports.

    Some MLX tensors, notably bf16-backed intermediates seen on Qwen3 + Quest,
    can fail under ``np.array(mx_tensor)`` with a PEP 3118 buffer mismatch. When
    a target NumPy dtype is known, cast through the corresponding MLX dtype
    first; otherwise fall back to ``tolist()`` as a correctness-first path.
    """
    target_dtype = np.dtype(dtype) if dtype is not None else None

    if isinstance(value, np.ndarray):
        array = value
    else:
        if target_dtype is not None and hasattr(value, "astype"):
            mx_dtype = _NUMPY_TO_MX_DTYPE.get(target_dtype)
            if mx_dtype is not None:
                value = value.astype(mx_dtype)
        try:
            mx.eval(value)
        except Exception:
            pass
        try:
            array = np.asarray(value)
        except (RuntimeError, TypeError, ValueError):
            if not hasattr(value, "tolist"):
                raise
            array = np.asarray(value.tolist())

    if target_dtype is not None and array.dtype != target_dtype:
        array = array.astype(target_dtype, copy=False)
    if copy:
        array = np.array(array, copy=True)
    return array


@dataclass
class MetalKvFrameSlot:
    frame_idx: int
    page_id: Optional[Tuple[int, int, int]] = None
    valid_length: int = 0
    version: int = -1
    buffer: Optional[mx.array] = None


class MetalKvFrameStorage:
    """Stable resident KV frame slots addressable by BufferPool frame id."""

    def __init__(self, num_frames: int, page_size: int, mx_dtype):
        self.page_size = int(page_size)
        self.mx_dtype = mx_dtype
        self.slots = [MetalKvFrameSlot(i) for i in range(int(num_frames))]
        self.buffers: List[Optional[mx.array]] = [None] * int(num_frames)
        self.versions: List[int] = [-1] * int(num_frames)

    def invalidate_frame(self, frame_idx: int):
        slot = self.slots[int(frame_idx)]
        slot.page_id = None
        slot.valid_length = 0
        slot.version = -1
        slot.buffer = None
        self.buffers[int(frame_idx)] = None
        self.versions[int(frame_idx)] = -1

    def bind_host_frame(
        self,
        frame_idx: int,
        page_id: Optional[Tuple[int, int, int]],
        host_frame: np.ndarray,
        version: int,
        *,
        valid_length: Optional[int] = None,
    ) -> mx.array:
        frame_idx = int(frame_idx)
        slot = self.slots[frame_idx]
        if (
            slot.buffer is not None
            and slot.page_id == page_id
            and slot.version == int(version)
        ):
            return slot.buffer
        buffer = mx.array(host_frame).astype(self.mx_dtype)
        return self.rebind_frame(
            frame_idx,
            page_id,
            buffer,
            version,
            valid_length=valid_length,
        )

    def ensure_frame(
        self,
        frame_idx: int,
        page_id: Optional[Tuple[int, int, int]],
        host_frame: np.ndarray,
        version: int,
        *,
        dtype=None,
        valid_length: Optional[int] = None,
    ) -> mx.array:
        frame_idx = int(frame_idx)
        slot = self.slots[frame_idx]
        if (
            slot.buffer is None
            or slot.page_id != page_id
            or slot.version != int(version)
        ):
            buffer = self.bind_host_frame(
                frame_idx,
                page_id,
                host_frame,
                version,
                valid_length=valid_length,
            )
        else:
            buffer = slot.buffer
        if dtype is not None and buffer.dtype != dtype:
            buffer = buffer.astype(dtype)
        return buffer

    def rebind_frame(
        self,
        frame_idx: int,
        page_id: Optional[Tuple[int, int, int]],
        buffer: mx.array,
        version: int,
        *,
        valid_length: Optional[int] = None,
    ) -> mx.array:
        frame_idx = int(frame_idx)
        slot = self.slots[frame_idx]
        slot.page_id = page_id
        slot.valid_length = self.page_size if valid_length is None else int(valid_length)
        slot.version = int(version)
        slot.buffer = buffer
        self.buffers[frame_idx] = buffer
        self.versions[frame_idx] = int(version)
        return buffer

    def sync_host_frame(
        self,
        frame_idx: int,
        host_frame: np.ndarray,
        host_version: int,
        dtype: np.dtype,
    ) -> int:
        frame_idx = int(frame_idx)
        slot = self.slots[frame_idx]
        if slot.buffer is None or slot.version == int(host_version):
            return int(host_version)
        host_frame[...] = safe_to_numpy(slot.buffer, dtype=dtype)
        return slot.version

    def slot_metadata(self, frame_idx: int) -> MetalKvFrameSlot:
        return self.slots[int(frame_idx)]


@dataclass
class EvictedPage:
    page_id: Any
    slot_id: int
    valid_length: int
    dirty: bool


class MetalKVBufferPoolMetadata:
    """Logical page to fixed resident Metal slot metadata."""

    def __init__(self, num_slots: int, page_size: int):
        self.num_slots = int(num_slots)
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        self.page_size = int(page_size)
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        self.page_to_slot: Dict[Any, int] = {}
        self.slot_to_page: List[Optional[Any]] = [None] * self.num_slots
        self.free_slots: List[int] = list(range(self.num_slots - 1, -1, -1))
        self.recent_pages: OrderedDict[Any, int] = OrderedDict()
        self.valid_length: Dict[Any, int] = {}
        self.dirty_recent_pages: Set[Any] = set()

    def has_page(self, page_id: Any) -> bool:
        return page_id in self.page_to_slot

    def get_slot(self, page_id: Any) -> Optional[int]:
        return self.page_to_slot.get(page_id)

    def touch(self, page_id: Any) -> None:
        if page_id in self.recent_pages:
            self.recent_pages.move_to_end(page_id, last=True)

    def allocate(
        self,
        page_id: Any,
        valid_length: Optional[int] = None,
        dirty: bool = False,
    ) -> tuple[int, Optional[EvictedPage]]:
        if page_id in self.page_to_slot:
            if valid_length is not None:
                self.set_valid_length(page_id, valid_length)
            if dirty:
                self.mark_dirty(page_id)
            self.touch(page_id)
            return self.page_to_slot[page_id], None

        length = self._normalize_valid_length(valid_length)
        if self.free_slots:
            slot_id = self.free_slots.pop()
            evicted = None
        else:
            if not self.recent_pages:
                raise RuntimeError("cannot allocate from an empty metadata pool")
            old_page_id, slot_id = self.recent_pages.popitem(last=False)
            evicted = self._remove_resident_page(old_page_id, slot_id)

        self.page_to_slot[page_id] = slot_id
        self.slot_to_page[slot_id] = page_id
        self.recent_pages[page_id] = slot_id
        self.valid_length[page_id] = length
        if dirty:
            self.dirty_recent_pages.add(page_id)
        return slot_id, evicted

    def evict(self, page_id: Any) -> Optional[EvictedPage]:
        slot_id = self.page_to_slot.get(page_id)
        if slot_id is None:
            return None
        evicted = self._remove_resident_page(page_id, slot_id)
        self.free_slots.append(slot_id)
        return evicted

    def mark_dirty(self, page_id: Any) -> None:
        if page_id in self.page_to_slot:
            self.dirty_recent_pages.add(page_id)

    def mark_clean(self, page_id: Any) -> None:
        self.dirty_recent_pages.discard(page_id)

    def set_valid_length(self, page_id: Any, length: int) -> None:
        if page_id not in self.page_to_slot:
            raise KeyError(page_id)
        self.valid_length[page_id] = self._normalize_valid_length(length)

    def get_valid_length(self, page_id: Any) -> int:
        if page_id not in self.page_to_slot:
            raise KeyError(page_id)
        return self.valid_length[page_id]

    def clear(self) -> None:
        self.page_to_slot.clear()
        self.slot_to_page = [None] * self.num_slots
        self.free_slots = list(range(self.num_slots - 1, -1, -1))
        self.recent_pages.clear()
        self.valid_length.clear()
        self.dirty_recent_pages.clear()

    def _remove_resident_page(self, page_id: Any, slot_id: int) -> EvictedPage:
        del self.page_to_slot[page_id]
        self.slot_to_page[slot_id] = None
        self.recent_pages.pop(page_id, None)
        valid_length = self.valid_length.pop(page_id, self.page_size)
        dirty = page_id in self.dirty_recent_pages
        self.dirty_recent_pages.discard(page_id)
        return EvictedPage(
            page_id=page_id,
            slot_id=int(slot_id),
            valid_length=int(valid_length),
            dirty=dirty,
        )

    def _normalize_valid_length(self, length: Optional[int]) -> int:
        value = self.page_size if length is None else int(length)
        if value < 0 or value > self.page_size:
            raise ValueError(
                f"valid_length {value} outside page bounds [0, {self.page_size}]"
            )
        return value


@dataclass
class ResidentFrameMeta:
    frame_id: int
    layer_idx: int = -1
    kv_head: int = -1
    block_idx: int = -1
    valid_length: int = 0
    version: int = 0
    dirty: bool = False


class ResidentKVSlotPool:
    """Fixed-slot KV storage for the Latest Resident KV path only."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        max_blocks: int,
        page_size: int,
        head_dim: int,
        mx_dtype,
        slot_blocks: Optional[int] = None,
    ):
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.max_blocks = int(max_blocks)
        self.slot_blocks = max(1, min(int(slot_blocks or max_blocks), self.max_blocks))
        self.page_size = int(page_size)
        self.head_dim = int(head_dim)
        self.mx_dtype = mx_dtype
        self.frame_shape = (2, self.page_size, self.head_dim)
        self.num_frames = self.num_layers * self.num_heads * self.slot_blocks
        self.frames: List[Optional[mx.array]] = [None] * self.num_frames
        self.layer_arenas: List[Optional[mx.array]] = [None] * self.num_layers
        self.layer_arena_blocks: List[int] = [0] * self.num_layers
        self.meta = [ResidentFrameMeta(i) for i in range(self.num_frames)]
        self.page_to_slot: List[Dict[int, int]] = [dict() for _ in range(self.num_layers)]
        self.recent_pages: List[OrderedDict] = [OrderedDict() for _ in range(self.num_layers)]
        self.decode_slot_blocks = min(2, self.slot_blocks)
        self.selected_slot_start = self.decode_slot_blocks if self.slot_blocks > 2 else 0
        self.free_slots: List[List[int]] = [
            list(range(self.slot_blocks - 1, self.selected_slot_start - 1, -1))
            for _ in range(self.num_layers)
        ]
        if not self.free_slots[0]:
            self.free_slots = [
                list(range(self.slot_blocks - 1, -1, -1)) for _ in range(self.num_layers)
            ]
        self.decode_page_to_slot: List[Dict[int, int]] = [
            dict() for _ in range(self.num_layers)
        ]
        self.decode_recent_pages: List[OrderedDict] = [
            OrderedDict() for _ in range(self.num_layers)
        ]
        self.decode_page_valid_lengths: List[Dict[int, int]] = [
            dict() for _ in range(self.num_layers)
        ]
        self.last_page_valid_length: List[int] = [0] * self.num_layers

    def frame_id(self, layer_idx: int, block_idx: int, kv_head: int) -> int:
        if block_idx < 0 or block_idx >= self.max_blocks:
            raise RuntimeError(
                f"resident frame block_idx {block_idx} outside capacity {self.max_blocks}"
            )
        slot_block = self.ensure_page_slot(layer_idx, block_idx)
        return (int(layer_idx) * self.slot_blocks + slot_block) * self.num_heads + int(kv_head)

    def frame_id_if_present(self, layer_idx: int, block_idx: int, kv_head: int) -> Optional[int]:
        slot_block = self.page_to_slot[int(layer_idx)].get(int(block_idx))
        if slot_block is None:
            return None
        return (int(layer_idx) * self.slot_blocks + slot_block) * self.num_heads + int(kv_head)

    def decode_frame_id(self, layer_idx: int, block_idx: int, kv_head: int) -> int:
        slot_block = self.ensure_decode_page_slot(layer_idx, block_idx)
        return (int(layer_idx) * self.slot_blocks + slot_block) * self.num_heads + int(kv_head)

    def decode_frame_id_if_present(
        self,
        layer_idx: int,
        block_idx: int,
        kv_head: int,
    ) -> Optional[int]:
        slot_block = self.decode_page_to_slot[int(layer_idx)].get(int(block_idx))
        if slot_block is None:
            return None
        return (int(layer_idx) * self.slot_blocks + slot_block) * self.num_heads + int(kv_head)

    def has_page(self, layer_idx: int, block_idx: int) -> bool:
        return int(block_idx) in self.page_to_slot[int(layer_idx)]

    def has_decode_page(self, layer_idx: int, block_idx: int) -> bool:
        return int(block_idx) in self.decode_page_to_slot[int(layer_idx)]

    def layer_local_frame_id(self, block_idx: int, kv_head: int) -> int:
        return int(block_idx) * self.num_heads + int(kv_head)

    def layer_local_frame_id_from_frame_id(self, layer_idx: int, frame_id: int) -> int:
        return int(frame_id) - (int(layer_idx) * self.slot_blocks * self.num_heads)

    def ensure_page_slot(self, layer_idx: int, block_idx: int) -> int:
        layer_idx = int(layer_idx)
        block_idx = int(block_idx)
        page_slots = self.page_to_slot[layer_idx]
        if block_idx in page_slots:
            self.recent_pages[layer_idx].move_to_end(block_idx, last=True)
            return page_slots[block_idx]
        if self.free_slots[layer_idx]:
            slot_block = self.free_slots[layer_idx].pop()
        else:
            old_block, slot_block = self.recent_pages[layer_idx].popitem(last=False)
            del page_slots[old_block]
        page_slots[block_idx] = slot_block
        self.recent_pages[layer_idx][block_idx] = slot_block
        return slot_block

    def ensure_decode_page_slot(self, layer_idx: int, block_idx: int) -> int:
        layer_idx = int(layer_idx)
        block_idx = int(block_idx)
        page_slots = self.decode_page_to_slot[layer_idx]
        if block_idx in page_slots:
            self.decode_recent_pages[layer_idx].move_to_end(block_idx, last=True)
            return page_slots[block_idx]

        used_slots = set(page_slots.values())
        for slot_block in range(self.decode_slot_blocks):
            if slot_block not in used_slots:
                break
        else:
            old_block, slot_block = self.decode_recent_pages[layer_idx].popitem(last=False)
            del page_slots[old_block]
            self.decode_page_valid_lengths[layer_idx].pop(old_block, None)

        page_slots[block_idx] = slot_block
        self.decode_recent_pages[layer_idx][block_idx] = slot_block
        return slot_block

    def ensure_frame(self, layer_idx: int, block_idx: int, kv_head: int) -> tuple[int, mx.array]:
        frame_id = self.frame_id(layer_idx, block_idx, kv_head)
        local_frame_id = self.layer_local_frame_id(block_idx, kv_head)
        frame = self.ensure_layer_arena(
            layer_idx,
            min_blocks=int(block_idx) + 1,
        )[local_frame_id]
        meta = self.meta[frame_id]
        meta.layer_idx = int(layer_idx)
        meta.kv_head = int(kv_head)
        meta.block_idx = int(block_idx)
        return frame_id, frame

    def ensure_layer_arena(self, layer_idx: int, dtype=None, min_blocks: int = 1) -> mx.array:
        target_dtype = dtype if dtype is not None else self.mx_dtype
        layer_idx = int(layer_idx)
        target_blocks = self.slot_blocks
        arena = self.layer_arenas[layer_idx]
        current_blocks = int(self.layer_arena_blocks[layer_idx])
        if arena is None or arena.dtype != target_dtype:
            arena = mx.zeros(
                (target_blocks * self.num_heads, 2, self.page_size, self.head_dim),
                dtype=target_dtype,
            )
            self.layer_arenas[layer_idx] = arena
            self.layer_arena_blocks[layer_idx] = target_blocks
        elif target_blocks > current_blocks:
            extra = mx.zeros(
                ((target_blocks - current_blocks) * self.num_heads, 2, self.page_size, self.head_dim),
                dtype=target_dtype,
            )
            arena = mx.concatenate([arena, extra], axis=0)
            self.layer_arenas[layer_idx] = arena
            self.layer_arena_blocks[layer_idx] = target_blocks
        return arena

    def write_kv_slices(
        self,
        layer_idx: int,
        block_idx: int,
        page_offset: int,
        k_mx: mx.array,
        v_mx: mx.array,
    ) -> List[int]:
        frame_ids = [
            self.frame_id(layer_idx, block_idx, kv_head)
            for kv_head in range(self.num_heads)
        ]
        return self.write_kv_slices_by_frame_ids(
            layer_idx,
            frame_ids,
            page_offset,
            k_mx,
            v_mx,
        )

    def write_page(self, layer_idx: int, block_idx: int, page_mx: mx.array) -> List[int]:
        frame_ids = [
            self.frame_id(layer_idx, block_idx, kv_head)
            for kv_head in range(self.num_heads)
        ]
        return self.write_kv_slices_by_frame_ids(
            layer_idx,
            frame_ids,
            0,
            page_mx[0],
            page_mx[1],
            dirty=False,
        )

    def write_decode_page(self, layer_idx: int, block_idx: int, page_mx: mx.array) -> List[int]:
        frame_ids = [
            self.decode_frame_id(layer_idx, block_idx, kv_head)
            for kv_head in range(self.num_heads)
        ]
        return self.write_kv_slices_by_frame_ids(
            layer_idx,
            frame_ids,
            0,
            page_mx[0],
            page_mx[1],
        )

    def write_decode_kv_slices(
        self,
        layer_idx: int,
        block_idx: int,
        page_offset: int,
        k_mx: mx.array,
        v_mx: mx.array,
    ) -> List[int]:
        frame_ids = [
            self.decode_frame_id(layer_idx, block_idx, kv_head)
            for kv_head in range(self.num_heads)
        ]
        written = self.write_kv_slices_by_frame_ids(
            layer_idx,
            frame_ids,
            page_offset,
            k_mx,
            v_mx,
        )
        end = int(page_offset) + int(k_mx.shape[0])
        self.decode_page_valid_lengths[int(layer_idx)][int(block_idx)] = end
        self.last_page_valid_length[int(layer_idx)] = end
        return written

    def write_kv_slices_by_frame_ids(
        self,
        layer_idx: int,
        frame_ids: List[int],
        page_offset: int,
        k_mx: mx.array,
        v_mx: mx.array,
        *,
        dirty: bool = True,
    ) -> List[int]:
        token_count = int(k_mx.shape[0])
        end = int(page_offset) + token_count
        frame_ids_np = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
        self.write_kv_slices_to_arena_by_frame_ids(
            layer_idx,
            frame_ids_np,
            page_offset,
            k_mx,
            v_mx,
            dtype=k_mx.dtype,
        )
        written = []
        for frame_id in frame_ids_np:
            frame_id = int(frame_id)
            local_frame_id = self.layer_local_frame_id_from_frame_id(layer_idx, frame_id)
            block_idx = local_frame_id // self.num_heads
            kv_head = local_frame_id % self.num_heads
            meta = self.meta[frame_id]
            meta.layer_idx = int(layer_idx)
            meta.kv_head = int(kv_head)
            meta.block_idx = int(block_idx)
            meta.valid_length = max(meta.valid_length, end)
            self.last_page_valid_length[int(layer_idx)] = end
            if dirty:
                meta.version += 1
            else:
                meta.version = 0
            meta.dirty = bool(dirty)
            written.append(frame_id)
        return written

    def write_kv_slices_to_arena_by_frame_ids(
        self,
        layer_idx: int,
        frame_ids: np.ndarray,
        page_offset: int,
        k_mx: mx.array,
        v_mx: mx.array,
        dtype=None,
    ) -> mx.array:
        frame_ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
        if frame_ids.shape[0] != int(k_mx.shape[1]):
            raise ValueError(
                f"frame_ids length {frame_ids.shape[0]} must match k_mx heads {k_mx.shape[1]}"
            )
        local_frame_ids = np.array(
            [
                self.layer_local_frame_id_from_frame_id(layer_idx, int(frame_id))
                for frame_id in frame_ids
            ],
            dtype=np.int64,
        )
        local_blocks = (local_frame_ids // self.num_heads) + 1
        arena = self.ensure_layer_arena(
            layer_idx,
            dtype=dtype,
            min_blocks=int(local_blocks.max()) if len(local_blocks) else 1,
        )
        native_out = resident_arena_write_batched(
            arena,
            mx.array(local_frame_ids),
            mx.contiguous(k_mx.astype(arena.dtype)),
            mx.contiguous(v_mx.astype(arena.dtype)),
            int(page_offset),
        )
        if native_out is None:
            raise RuntimeError("resident arena write primitive unavailable")
        arena = native_out[0]
        self.layer_arenas[int(layer_idx)] = arena
        return arena

    def get_selected_frame_arena_with_last(
        self,
        layer_idx: int,
        frame_indices: np.ndarray,
        kv_head_indices: np.ndarray,
        last_block_idx: int,
        dtype=None,
    ) -> Tuple[mx.array, mx.array]:
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        num_heads, selected_count = frame_indices.shape
        kv_head_indices = np.asarray(kv_head_indices, dtype=np.int64).reshape(num_heads)
        selected_frames = np.empty((num_heads, selected_count + 1), dtype=np.int32)
        for h in range(num_heads):
            kv_head = int(kv_head_indices[h])
            for j, frame_idx in enumerate(frame_indices[h]):
                selected_frames[h, j] = self.layer_local_frame_id(
                    int(frame_idx),
                    kv_head,
                )
            selected_frames[h, selected_count] = self.layer_local_frame_id(
                int(last_block_idx),
                kv_head,
            )
        return mx.array(selected_frames), self.get_layer_frame_arena(layer_idx, dtype=dtype)

    def get_selected_frame_arena_with_last_by_frame_ids(
        self,
        layer_idx: int,
        frame_ids: np.ndarray,
        last_frame_ids: np.ndarray,
        dtype=None,
    ) -> Tuple[mx.array, mx.array]:
        frame_ids = np.asarray(frame_ids, dtype=np.int64)
        last_frame_ids = np.asarray(last_frame_ids, dtype=np.int64).reshape(frame_ids.shape[0])
        num_heads, selected_count = frame_ids.shape
        selected_frames = np.empty((num_heads, selected_count + 1), dtype=np.int32)
        for h in range(num_heads):
            for j, frame_id in enumerate(frame_ids[h]):
                selected_frames[h, j] = self.layer_local_frame_id_from_frame_id(
                    layer_idx,
                    int(frame_id),
                )
            selected_frames[h, selected_count] = self.layer_local_frame_id_from_frame_id(
                layer_idx,
                int(last_frame_ids[h]),
            )
        return mx.array(selected_frames), self.get_layer_frame_arena(layer_idx, dtype=dtype)

    def get_selected_frame_buffers_with_last_by_frame_ids(
        self,
        layer_idx: int,
        frame_ids: np.ndarray,
        last_frame_ids: np.ndarray,
        dtype=None,
    ) -> Tuple[mx.array, mx.array]:
        return self.get_selected_frame_arena_with_last_by_frame_ids_stable(
            layer_idx,
            frame_ids,
            last_frame_ids,
            dtype=dtype,
        )

    def get_selected_frame_arena_with_last_by_frame_ids_stable(
        self,
        layer_idx: int,
        frame_ids: np.ndarray,
        last_frame_ids: np.ndarray,
        dtype=None,
    ) -> Tuple[mx.array, mx.array]:
        frame_ids = np.asarray(frame_ids, dtype=np.int64)
        last_frame_ids = np.asarray(last_frame_ids, dtype=np.int64).reshape(frame_ids.shape[0])
        num_heads, selected_count = frame_ids.shape
        selected_frames = np.empty((num_heads, selected_count + 1), dtype=np.int32)
        for h in range(num_heads):
            for j, frame_id in enumerate(frame_ids[h]):
                selected_frames[h, j] = self.layer_local_frame_id_from_frame_id(
                    layer_idx,
                    int(frame_id),
                )
            selected_frames[h, selected_count] = self.layer_local_frame_id_from_frame_id(
                layer_idx,
                int(last_frame_ids[h]),
            )
        max_local = 0
        if selected_frames.size:
            max_local = int(selected_frames.max())
        arena = self.ensure_layer_arena(
            layer_idx,
            dtype=dtype,
            min_blocks=max_local // self.num_heads + 1,
        )
        return mx.array(selected_frames), arena

    def get_layer_frame_arena(self, layer_idx: int, dtype=None) -> mx.array:
        return self.ensure_layer_arena(layer_idx, dtype=dtype)


ResidentFramePool = ResidentKVSlotPool


class BufferPool:
    """
    User-space buffer pool for head-sliced KV pages.
    Each page is (2, page_size, head_dim) for a single KV head.
    """
    def __init__(
        self,
        backing_file: str,
        num_layers: int,
        capacity: int,
        num_heads: int,
        page_size: int,
        head_dim: int,
        dtype: np.dtype,
        mx_dtype,
        num_frames: int,
        async_write: bool = False,
        disable_os_cache: bool = True,
        on_evict: Optional[Callable[[Tuple[int, int, int]], None]] = None,
    ):
        self.backing_file = backing_file
        self.num_layers = num_layers
        self.capacity = capacity
        self.num_heads = num_heads
        self.page_size = page_size
        self.head_dim = head_dim
        self.dtype = np.dtype(dtype)
        self.mx_dtype = mx_dtype
        self.frame_shape = (2, page_size, head_dim)
        self.page_bytes = int(np.prod(self.frame_shape) * self.dtype.itemsize)
        self.total_pages = num_layers * capacity * num_heads
        self.total_bytes = self.total_pages * self.page_bytes
        self.num_frames = max(1, num_frames)

        backing_dir = os.path.dirname(backing_file)
        if backing_dir:
            os.makedirs(backing_dir, exist_ok=True)
        self.fd = os.open(backing_file, os.O_RDWR | os.O_CREAT)
        os.ftruncate(self.fd, self.total_bytes)
        if disable_os_cache and fcntl is not None:
            try:
                fcntl.fcntl(self.fd, fcntl.F_NOCACHE, 1)
            except Exception:
                pass

        self.frames = [
            np.zeros(self.frame_shape, dtype=self.dtype)
            for _ in range(self.num_frames)
        ]
        self.frame_dirty = [False] * self.num_frames
        self.frame_page_id: List[Optional[Tuple[int, int, int]]] = [None] * self.num_frames
        self.frame_mx_cache: List[Optional[mx.array]] = [None] * self.num_frames
        self.frame_version = [0] * self.num_frames
        self.frame_host_versions = [0] * self.num_frames
        self.metal_kv_storage = MetalKvFrameStorage(
            self.num_frames,
            self.page_size,
            self.mx_dtype,
        )
        self.resident_frames_mx: Optional[List[Optional[mx.array]]] = (
            self.metal_kv_storage.buffers
        )
        self.resident_frame_versions: Optional[List[int]] = (
            self.metal_kv_storage.versions
        )
        self._resident_storage_enabled = False
        self.layer_frames_mx_cache: List[Optional[mx.array]] = [
            None for _ in range(self.num_layers)
        ]
        self.layer_frames_mx_version_cache: List[Optional[List[int]]] = [
            None for _ in range(self.num_layers)
        ]
        self.hot_frames_mx_cache: List[Optional[mx.array]] = [
            None for _ in range(self.num_layers)
        ]
        self.hot_frame_versions: List[Optional[List[int]]] = [
            None for _ in range(self.num_layers)
        ]
        self.hot_frame_keys: List[Optional[List[Optional[Tuple[int, int, int]]]]] = [
            None for _ in range(self.num_layers)
        ]
        self.hot_frame_lru: List[OrderedDict] = [
            OrderedDict() for _ in range(self.num_layers)
        ]
        self.hot_frame_free: List[Optional[List[int]]] = [
            None for _ in range(self.num_layers)
        ]
        self.hot_frame_capacity: List[int] = []

        self.page_table: Dict[Tuple[int, int, int], int] = {}
        # Partition LRU by layer to avoid cross-layer eviction.
        if self.num_frames < self.num_layers:
            raise ValueError(
                "num_frames must be >= num_layers for layer-partitioned LRU"
            )
        frames_per_layer = self.num_frames // self.num_layers
        remainder = self.num_frames % self.num_layers
        self._layer_frame_offsets: List[int] = []
        self._layer_frame_counts: List[int] = []
        self.lru: List[OrderedDict] = []
        self.free_list: List[List[int]] = []
        base = 0
        for layer_idx in range(self.num_layers):
            count = frames_per_layer + (1 if layer_idx < remainder else 0)
            self._layer_frame_offsets.append(base)
            self._layer_frame_counts.append(count)
            self.lru.append(OrderedDict())
            self.free_list.append(list(range(base, base + count)))
            base += count
        hot_capacity_limit = int(os.environ.get("ALAYAJET_QUEST_HOT_FRAME_CAPACITY", "256"))
        self.hot_frame_capacity = [
            max(1, min(count, hot_capacity_limit))
            for count in self._layer_frame_counts
        ]

        self.read_bytes = 0
        self.write_bytes = 0
        self.hits = 0
        self.misses = 0
        self.selected_hits = 0
        self.selected_misses = 0
        self.selected_requests = 0

        self._write_queue: Optional[queue.Queue] = None
        self._write_thread: Optional[threading.Thread] = None
        self.async_write = async_write
        self._on_evict = on_evict
        self.selected_hot: List[OrderedDict] = [OrderedDict() for _ in range(self.num_layers)]
        self._selected_hot_capacity = [
            max(1, count // 2) for count in self._layer_frame_counts
        ]
        if async_write:
            self._write_queue = queue.Queue()
            self._write_thread = threading.Thread(
                target=self._write_worker,
                daemon=True,
            )
            self._write_thread.start()

    def _page_offset(self, page_id: Tuple[int, int, int]) -> int:
        layer_idx, page_idx, kv_head = page_id
        logical_index = (layer_idx * self.capacity + page_idx) * self.num_heads + kv_head
        return logical_index * self.page_bytes

    def _read_page(self, page_id: Tuple[int, int, int], frame: np.ndarray):
        offset = self._page_offset(page_id)
        data = os.pread(self.fd, self.page_bytes, offset)
        if len(data) != self.page_bytes:
            raise RuntimeError(
                f"Short read for page {page_id}: {len(data)} != {self.page_bytes}"
            )
        frame_view = np.frombuffer(data, dtype=self.dtype).reshape(self.frame_shape)
        frame[...] = frame_view
        self.read_bytes += self.page_bytes

    def _read_pages_all_heads(
        self,
        layer_idx: int,
        page_start: int,
        page_count: int,
    ) -> np.ndarray:
        if page_count <= 0:
            return np.empty((0, self.num_heads, *self.frame_shape), dtype=self.dtype)
        offset = self._page_offset((layer_idx, page_start, 0))
        total_bytes = page_count * self.num_heads * self.page_bytes
        data = os.pread(self.fd, total_bytes, offset)
        if len(data) != total_bytes:
            raise RuntimeError(
                f"Short read for pages {page_start}:{page_start + page_count} "
                f"(layer {layer_idx}): {len(data)} != {total_bytes}"
            )
        self.read_bytes += total_bytes
        return np.frombuffer(data, dtype=self.dtype).reshape(
            page_count, self.num_heads, *self.frame_shape
        )

    def _write_page(self, page_id: Tuple[int, int, int], frame: np.ndarray):
        offset = self._page_offset(page_id)
        written = os.pwrite(self.fd, frame, offset)
        if written != self.page_bytes:
            raise RuntimeError(
                f"Short write for page {page_id}: {written} != {self.page_bytes}"
            )
        self.write_bytes += self.page_bytes

    def _invalidate_page(self, page_id: Tuple[int, int, int]):
        frame_idx = self.page_table.get(page_id)
        if frame_idx is None:
            return
        layer_idx = page_id[0]
        layer_lru = self.lru[layer_idx]
        if page_id in layer_lru:
            del layer_lru[page_id]
        del self.page_table[page_id]
        self.frame_dirty[frame_idx] = False
        self.frame_page_id[frame_idx] = None
        self.frame_mx_cache[frame_idx] = None
        self.frame_host_versions[frame_idx] = -1
        self.metal_kv_storage.invalidate_frame(frame_idx)
        self.free_list[layer_idx].append(frame_idx)
        self.selected_hot[layer_idx].pop(page_id, None)
        self._invalidate_hot_frame_slot(page_id)
        if self._on_evict is not None:
            self._on_evict(page_id)

    def write_page_direct(self, page_id: Tuple[int, int, int], frame: np.ndarray):
        frame_np = np.ascontiguousarray(frame, dtype=self.dtype)
        self._enqueue_write(page_id, frame_np)
        self._invalidate_page(page_id)

    def _enqueue_write(self, page_id: Tuple[int, int, int], frame: np.ndarray):
        if self._write_queue is None:
            self._write_page(page_id, frame)
            return
        self._write_queue.put(("page", page_id, np.array(frame, copy=True)))

    def _enqueue_raw_write(self, offset: int, data: np.ndarray):
        if self._write_queue is None:
            written = os.pwrite(self.fd, data, offset)
            if written != data.nbytes:
                raise RuntimeError(
                    f"Short write at offset {offset}: {written} != {data.nbytes}"
                )
            self.write_bytes += data.nbytes
            return
        self._write_queue.put(("raw", offset, data, data.nbytes))

    def _write_worker(self):
        if self._write_queue is None:
            return
        while True:
            item = self._write_queue.get()
            if item is None:
                self._write_queue.task_done()
                break
            kind = item[0]
            if kind == "page":
                _, page_id, frame = item
                self._write_page(page_id, frame)
            elif kind == "raw":
                _, offset, data, size = item
                written = os.pwrite(self.fd, data, offset)
                if written != size:
                    raise RuntimeError(
                        f"Short write at offset {offset}: {written} != {size}"
                    )
                self.write_bytes += size
            self._write_queue.task_done()

    def _evict_frame(self, layer_idx: int) -> int:
        layer_lru = self.lru[layer_idx]
        if not layer_lru:
            raise RuntimeError(f"LRU empty for layer {layer_idx}; cannot evict")
        hot = self.selected_hot[layer_idx]
        page_id = None
        frame_idx = None
        for candidate_page_id, candidate_frame_idx in layer_lru.items():
            if candidate_page_id not in hot:
                page_id = candidate_page_id
                frame_idx = candidate_frame_idx
                break
        if page_id is None:
            page_id, frame_idx = layer_lru.popitem(last=False)
        else:
            del layer_lru[page_id]
        hot.pop(page_id, None)
        if self._on_evict is not None:
            self._on_evict(page_id)
        if self.frame_dirty[frame_idx]:
            self._sync_host_frame_from_resident(frame_idx)
            self._enqueue_write(page_id, self.frames[frame_idx])
            self.frame_dirty[frame_idx] = False
        del self.page_table[page_id]
        self.frame_page_id[frame_idx] = None
        self.frame_mx_cache[frame_idx] = None
        self.frame_host_versions[frame_idx] = -1
        self.metal_kv_storage.invalidate_frame(frame_idx)
        self._invalidate_hot_frame_slot(page_id)
        return frame_idx

    def _mark_frame_changed(self, layer_idx: int, frame_idx: int, *, host_current: bool = True):
        self.frame_version[frame_idx] += 1
        self.frame_mx_cache[frame_idx] = None
        if host_current:
            self.frame_host_versions[frame_idx] = self.frame_version[frame_idx]
        self._update_layer_frame_cache(layer_idx, frame_idx)
        page_id = self.frame_page_id[frame_idx]
        if page_id is not None:
            self._update_hot_frame_slot(page_id, frame_idx)
        if host_current:
            self._update_resident_frame_cache(frame_idx)

    def _alloc_frame(self, layer_idx: int) -> int:
        layer_free = self.free_list[layer_idx]
        if layer_free:
            return layer_free.pop()
        return self._evict_frame(layer_idx)

    def _touch(self, page_id: Tuple[int, int, int]):
        layer_lru = self.lru[page_id[0]]
        if page_id in layer_lru:
            layer_lru.move_to_end(page_id, last=True)

    def mark_selected_pages(self, page_ids: List[Tuple[int, int, int]]):
        if not page_ids:
            return
        unique_page_ids = list(dict.fromkeys(page_ids))
        self.selected_requests += len(unique_page_ids)
        for page_id in unique_page_ids:
            layer_idx = page_id[0]
            if page_id in self.page_table:
                self.selected_hits += 1
            else:
                self.selected_misses += 1
            hot = self.selected_hot[layer_idx]
            hot[page_id] = None
            hot.move_to_end(page_id, last=True)
            capacity = self._selected_hot_capacity[layer_idx]
            while len(hot) > capacity:
                hot.popitem(last=False)

    def has_page(self, page_id: Tuple[int, int, int]) -> bool:
        return page_id in self.page_table

    def _get_frame(self, page_id: Tuple[int, int, int], assume_zero: bool) -> int:
        frame_idx = self.page_table.get(page_id)
        if frame_idx is not None:
            self.hits += 1
            self._touch(page_id)
            return frame_idx
        self.misses += 1
        layer_idx = page_id[0]
        frame_idx = self._alloc_frame(layer_idx)
        self.page_table[page_id] = frame_idx
        self.frame_page_id[frame_idx] = page_id
        self.lru[layer_idx][page_id] = frame_idx
        if assume_zero:
            self.frames[frame_idx].fill(0)
        else:
            self._read_page(page_id, self.frames[frame_idx])
        self._mark_frame_changed(layer_idx, frame_idx)
        return frame_idx

    def get_page(self, page_id: Tuple[int, int, int], assume_zero: bool = False) -> np.ndarray:
        frame_idx = self._get_frame(page_id, assume_zero)
        return self.frames[frame_idx]

    def get_page_mx(self, page_id: Tuple[int, int, int], assume_zero: bool = False) -> mx.array:
        frame_idx = self._get_frame(page_id, assume_zero)
        return self._frame_mx_for_cache(frame_idx)

    def get_frame_index(self, page_id: Tuple[int, int, int], assume_zero: bool = False) -> int:
        """Return the resident frame index for ``page_id``, loading it if needed."""
        return self._get_frame(page_id, assume_zero)

    def _ensure_resident_frames_mx(self):
        self._resident_storage_enabled = True
        self.resident_frames_mx = self.metal_kv_storage.buffers
        self.resident_frame_versions = self.metal_kv_storage.versions
        return self.resident_frames_mx

    def _update_resident_frame_cache(self, frame_idx: int):
        if not self._resident_storage_enabled:
            return
        self._ensure_resident_frames_mx()
        self.metal_kv_storage.bind_host_frame(
            frame_idx,
            self.frame_page_id[frame_idx],
            self.frames[frame_idx],
            self.frame_version[frame_idx],
        )

    def _frame_mx_for_cache(self, frame_idx: int) -> mx.array:
        cached = self.frame_mx_cache[frame_idx]
        if cached is not None:
            return cached
        resident = self.metal_kv_storage.slot_metadata(frame_idx).buffer
        if resident is not None:
            cached = resident
        else:
            cached = mx.array(self.frames[frame_idx]).astype(self.mx_dtype)
        self.frame_mx_cache[frame_idx] = cached
        return cached

    def _sync_host_frame_from_resident(self, frame_idx: int):
        if self.frame_host_versions[frame_idx] == self.frame_version[frame_idx]:
            return
        self.frame_host_versions[frame_idx] = self.metal_kv_storage.sync_host_frame(
            frame_idx,
            self.frames[frame_idx],
            self.frame_host_versions[frame_idx],
            self.dtype,
        )

    def _write_resident_frame_rows(
        self,
        frame_idx: int,
        page_offset: int,
        k_slice: mx.array,
        v_slice: mx.array,
    ):
        """Update resident arena through flat token rows and rebind it.

        MLX slicing creates copy-like graph updates, and writing a large
        multidimensional resident arena with ``arena[frame, kv, offset:end]``
        made one-token cache writes scale with arena size.  Flattening to token
        rows keeps the indexed update at the cache slot granularity.  The
        rebind mirrors vllm-metal's cache-write pattern: subsequent attention
        kernels see the updated array identity instead of an older resident
        arena object.
        """
        token_count = int(k_slice.shape[0])
        end = page_offset + token_count
        if page_offset < 0 or end > self.page_size:
            raise ValueError(
                f"resident KV write [{page_offset}:{end}] exceeds page_size={self.page_size}"
            )
        k_slice = mx.contiguous(k_slice.astype(self.mx_dtype))
        v_slice = mx.contiguous(v_slice.astype(self.mx_dtype))
        self._ensure_resident_frames_mx()
        slot = self.metal_kv_storage.slot_metadata(frame_idx)
        resident_frame = self.metal_kv_storage.ensure_frame(
            frame_idx,
            self.frame_page_id[frame_idx],
            self.frames[frame_idx],
            self.frame_version[frame_idx],
            valid_length=slot.valid_length,
        )
        native_out = native_resident_write(
            resident_frame,
            k_slice,
            v_slice,
            int(page_offset),
        )
        if native_out is not None:
            resident_frame = native_out[0]
        else:
            resident_frame = mx.array(resident_frame)
            resident_frame[0, page_offset:end] = k_slice
            resident_frame[1, page_offset:end] = v_slice
        return resident_frame

    def get_resident_frames_mx(
        self,
        required_frame_indices: Optional[List[int]] = None,
        dtype=None,
    ) -> mx.array:
        """Return a stacked resident frame arena for compatibility."""
        target_dtype = dtype if dtype is not None else self.mx_dtype
        frames = self.get_resident_frame_buffers_mx(
            required_frame_indices=required_frame_indices,
            dtype=target_dtype,
        )
        if not frames:
            return mx.zeros((0, *self.frame_shape), dtype=target_dtype)
        arena = mx.stack(frames, axis=0)
        if arena.dtype != target_dtype:
            arena = arena.astype(target_dtype)
        return arena

    def get_resident_frame_buffers_mx(
        self,
        required_frame_indices: Optional[List[int]] = None,
        dtype=None,
    ) -> List[mx.array]:
        """Return resident frame buffers without stacking them into an arena."""
        self._ensure_resident_frames_mx()
        if required_frame_indices is None:
            frame_indices = [
                idx
                for idx, page_id in enumerate(self.frame_page_id)
                if page_id is not None
            ]
        else:
            frame_indices = list(dict.fromkeys(int(idx) for idx in required_frame_indices))
        target_dtype = dtype if dtype is not None else self.mx_dtype
        buffers = []
        for frame_idx in frame_indices:
            frame = self.metal_kv_storage.ensure_frame(
                frame_idx,
                self.frame_page_id[frame_idx],
                self.frames[frame_idx],
                self.frame_version[frame_idx],
                dtype=target_dtype,
            )
            buffers.append(frame)
        return buffers

    def get_resident_layer_slot_buffers_mx(
        self,
        layer_idx: int,
        required_frame_indices: List[int],
        dtype=None,
    ) -> List[mx.array]:
        """Return resident buffers ordered by stable layer-local frame slot id."""
        if not required_frame_indices:
            return []
        self._ensure_resident_frames_mx()
        target_dtype = dtype if dtype is not None else self.mx_dtype
        local_indices = [
            self.layer_local_frame_index(layer_idx, frame_idx)
            for frame_idx in required_frame_indices
        ]
        max_local_idx = max(local_indices)
        required_frame_set = set(int(idx) for idx in required_frame_indices)
        zero = None
        buffers: List[mx.array] = []
        for local_idx in range(max_local_idx + 1):
            frame_idx = self._layer_frame_offsets[layer_idx] + local_idx
            if frame_idx in required_frame_set:
                buffers.append(
                    self.metal_kv_storage.ensure_frame(
                        frame_idx,
                        self.frame_page_id[frame_idx],
                        self.frames[frame_idx],
                        self.frame_version[frame_idx],
                        dtype=target_dtype,
                    )
                )
            else:
                if zero is None:
                    zero = mx.zeros(self.frame_shape, dtype=target_dtype)
                buffers.append(zero)
        return buffers

    def write_kv_slice_mx_resident(
        self,
        page_id: Tuple[int, int, int],
        page_offset: int,
        k_slice: mx.array,
        v_slice: mx.array,
        assume_zero: bool,
    ) -> int:
        """Write a K/V slice directly into the MLX resident arena.

        The NumPy frame and disk backing remain the correctness fallback, but
        the Metal selected-frame path can read the resident arena by frame id.
        """
        frame_idx = self._get_frame(page_id, assume_zero=assume_zero)
        resident_frame = self._write_resident_frame_rows(
            frame_idx,
            page_offset,
            k_slice,
            v_slice,
        )
        self.frame_version[frame_idx] += 1
        valid_length = max(
            self.metal_kv_storage.slot_metadata(frame_idx).valid_length,
            int(page_offset) + int(k_slice.shape[0]),
        )
        self.metal_kv_storage.rebind_frame(
            frame_idx,
            page_id,
            resident_frame,
            self.frame_version[frame_idx],
            valid_length=valid_length,
        )
        self.frame_dirty[frame_idx] = True
        self.frame_mx_cache[frame_idx] = None
        return frame_idx

    def write_kv_slices_mx_resident_batched(
        self,
        page_ids: List[Tuple[int, int, int]],
        page_offset: int,
        k_mx: mx.array,
        v_mx: mx.array,
        assume_zero: bool,
    ) -> List[int]:
        """Write one layer's KV heads with one native launch when available."""
        if not page_ids:
            return []
        token_count = int(k_mx.shape[0])
        end = int(page_offset) + token_count
        if page_offset < 0 or end > self.page_size:
            raise ValueError(
                f"resident KV batched write [{page_offset}:{end}] exceeds page_size={self.page_size}"
            )
        frame_indices = [
            self._get_frame(page_id, assume_zero=assume_zero)
            for page_id in page_ids
        ]
        self._ensure_resident_frames_mx()
        frames = [
            self.metal_kv_storage.ensure_frame(
                frame_idx,
                self.frame_page_id[frame_idx],
                self.frames[frame_idx],
                self.frame_version[frame_idx],
                valid_length=self.metal_kv_storage.slot_metadata(frame_idx).valid_length,
            )
            for frame_idx in frame_indices
        ]
        k_mx = mx.contiguous(k_mx.astype(self.mx_dtype))
        v_mx = mx.contiguous(v_mx.astype(self.mx_dtype))
        native_out = native_resident_write_batched(frames, k_mx, v_mx, int(page_offset))
        if native_out is None:
            for i, frame_idx in enumerate(frame_indices):
                frames[i] = self._write_resident_frame_rows(
                    frame_idx,
                    page_offset,
                    k_mx[:, i, :],
                    v_mx[:, i, :],
                )
        else:
            frames = list(native_out)
        valid_length = end
        for page_id, frame_idx, resident_frame in zip(page_ids, frame_indices, frames):
            self.frame_version[frame_idx] += 1
            valid_length = max(
                self.metal_kv_storage.slot_metadata(frame_idx).valid_length,
                end,
            )
            self.metal_kv_storage.rebind_frame(
                frame_idx,
                page_id,
                resident_frame,
                self.frame_version[frame_idx],
                valid_length=valid_length,
            )
            self.frame_dirty[frame_idx] = True
            self.frame_mx_cache[frame_idx] = None
        return frame_indices

    def get_frames_mx(self, frame_indices: List[int], dtype=None) -> mx.array:
        """Return a compact MLX frame arena for the requested resident frames."""
        if not frame_indices:
            return mx.array([])
        frames = []
        target_dtype = dtype if dtype is not None else self.mx_dtype
        for frame_idx in frame_indices:
            cached = self._frame_mx_for_cache(frame_idx)
            if cached.dtype != target_dtype:
                cached = cached.astype(target_dtype)
            frames.append(cached)
        return mx.stack(frames, axis=0)

    def get_frame_buffers_mx(self, frame_indices: List[int], dtype=None) -> List[mx.array]:
        """Return selected frame buffers without stacking them into an arena."""
        target_dtype = dtype if dtype is not None else self.mx_dtype
        buffers = []
        for frame_idx in frame_indices:
            cached = self._frame_mx_for_cache(frame_idx)
            if cached.dtype != target_dtype:
                cached = cached.astype(target_dtype)
            buffers.append(cached)
        return buffers

    def get_layer_frames_mx(
        self,
        layer_idx: int,
        dtype=None,
        required_frame_indices: Optional[List[int]] = None,
    ) -> mx.array:
        """Return a cached MLX arena for resident frame slots in one layer.

        When ``required_frame_indices`` is provided, only those slots are
        synchronized. This keeps the experimental layer-local direct-index path
        from uploading every frame in a large layer arena on first use.
        """
        cached = self.layer_frames_mx_cache[layer_idx]
        target_dtype = dtype if dtype is not None else self.mx_dtype
        if cached is None:
            start = self._layer_frame_offsets[layer_idx]
            count = self._layer_frame_counts[layer_idx]
            cached = mx.zeros((count, *self.frame_shape), dtype=self.mx_dtype)
            self.layer_frames_mx_cache[layer_idx] = cached
            self.layer_frames_mx_version_cache[layer_idx] = [-1] * count
        versions = self.layer_frames_mx_version_cache[layer_idx]
        if versions is not None:
            if required_frame_indices is None:
                start = self._layer_frame_offsets[layer_idx]
                count = self._layer_frame_counts[layer_idx]
                frame_indices = [
                    frame_idx
                    for frame_idx in range(start, start + count)
                    if self.frame_page_id[frame_idx] is not None
                ]
            else:
                frame_indices = list(dict.fromkeys(int(idx) for idx in required_frame_indices))
            for frame_idx in frame_indices:
                local_idx = self.layer_local_frame_index(layer_idx, frame_idx)
                if local_idx < 0 or local_idx >= len(versions):
                    continue
                if versions[local_idx] == self.frame_version[frame_idx]:
                    continue
                cached[local_idx] = self._frame_mx_for_cache(frame_idx)
                versions[local_idx] = self.frame_version[frame_idx]
        if cached.dtype != target_dtype:
            cached = cached.astype(target_dtype)
        return cached

    def _ensure_hot_frame_arena(self, layer_idx: int):
        cached = self.hot_frames_mx_cache[layer_idx]
        if cached is None:
            capacity = self.hot_frame_capacity[layer_idx]
            cached = mx.zeros((capacity, *self.frame_shape), dtype=self.mx_dtype)
            self.hot_frames_mx_cache[layer_idx] = cached
            self.hot_frame_versions[layer_idx] = [-1] * capacity
            self.hot_frame_keys[layer_idx] = [None] * capacity
            self.hot_frame_free[layer_idx] = list(range(capacity - 1, -1, -1))
        return cached

    def _invalidate_hot_frame_slot(self, page_id: Tuple[int, int, int]):
        layer_idx = page_id[0]
        hot = self.hot_frame_lru[layer_idx]
        slot = hot.pop(page_id, None)
        if slot is None:
            return
        keys = self.hot_frame_keys[layer_idx]
        versions = self.hot_frame_versions[layer_idx]
        free = self.hot_frame_free[layer_idx]
        if keys is not None:
            keys[slot] = None
        if versions is not None:
            versions[slot] = -1
        if free is not None:
            free.append(slot)

    def _update_hot_frame_slot(self, page_id: Tuple[int, int, int], frame_idx: int):
        layer_idx = page_id[0]
        hot = self.hot_frame_lru[layer_idx]
        slot = hot.get(page_id)
        if slot is None:
            return
        versions = self.hot_frame_versions[layer_idx]
        cached = self.hot_frames_mx_cache[layer_idx]
        if versions is None or cached is None:
            return
        if versions[slot] == self.frame_version[frame_idx]:
            return
        cached[slot] = mx.array(self.frames[frame_idx]).astype(self.mx_dtype)
        versions[slot] = self.frame_version[frame_idx]

    def get_hot_selected_frames_mx(
        self,
        layer_idx: int,
        page_ids: List[Tuple[int, int, int]],
        frame_indices: List[int],
        dtype=None,
    ) -> Tuple[np.ndarray, mx.array]:
        """Return selected hot-arena slot ids plus a small persistent arena."""
        if len(page_ids) != len(frame_indices):
            raise ValueError("page_ids and frame_indices must have the same length")
        cached = self._ensure_hot_frame_arena(layer_idx)
        versions = self.hot_frame_versions[layer_idx]
        keys = self.hot_frame_keys[layer_idx]
        free = self.hot_frame_free[layer_idx]
        hot = self.hot_frame_lru[layer_idx]
        if versions is None or keys is None or free is None:
            raise RuntimeError("hot frame arena is not initialized")

        slots = np.empty((len(page_ids),), dtype=np.int32)
        for i, (page_id, frame_idx) in enumerate(zip(page_ids, frame_indices)):
            slot = hot.get(page_id)
            if slot is None:
                if free:
                    slot = free.pop()
                else:
                    evicted_page_id, slot = hot.popitem(last=False)
                    keys[slot] = None
                    versions[slot] = -1
                hot[page_id] = slot
                keys[slot] = page_id
            else:
                hot.move_to_end(page_id, last=True)
            if versions[slot] != self.frame_version[frame_idx]:
                cached[slot] = self._frame_mx_for_cache(frame_idx)
                versions[slot] = self.frame_version[frame_idx]
            slots[i] = int(slot)

        target_dtype = dtype if dtype is not None else self.mx_dtype
        if cached.dtype != target_dtype:
            cached = cached.astype(target_dtype)
        return slots, cached

    def layer_local_frame_index(self, layer_idx: int, frame_idx: int) -> int:
        return int(frame_idx) - self._layer_frame_offsets[layer_idx]

    def _update_layer_frame_cache(self, layer_idx: int, frame_idx: int):
        cached = self.layer_frames_mx_cache[layer_idx]
        versions = self.layer_frames_mx_version_cache[layer_idx]
        if cached is None:
            return
        local_idx = self.layer_local_frame_index(layer_idx, frame_idx)
        if local_idx < 0 or local_idx >= cached.shape[0]:
            return
        cached[local_idx] = self._frame_mx_for_cache(frame_idx)
        if versions is not None and local_idx < len(versions):
            versions[local_idx] = self.frame_version[frame_idx]

    def read_page_raw(self, page_id: Tuple[int, int, int]) -> np.ndarray:
        offset = self._page_offset(page_id)
        data = os.pread(self.fd, self.page_bytes, offset)
        if len(data) != self.page_bytes:
            raise RuntimeError(
                f"Short read for page {page_id}: {len(data)} != {self.page_bytes}"
            )
        return np.frombuffer(data, dtype=self.dtype).reshape(self.frame_shape)

    def insert_page_cache(self, page_id: Tuple[int, int, int], frame: np.ndarray):
        frame_idx = self._get_frame(page_id, assume_zero=True)
        self.frames[frame_idx][...] = frame
        self.frame_dirty[frame_idx] = False
        self._mark_frame_changed(page_id[0], frame_idx)

    def write_page(self, page_id: Tuple[int, int, int], data: np.ndarray):
        frame_idx = self._get_frame(page_id, assume_zero=True)
        self.frames[frame_idx][...] = data
        self.frame_dirty[frame_idx] = True
        self._mark_frame_changed(page_id[0], frame_idx)

    def write_page_async(self, page_id: Tuple[int, int, int], data: np.ndarray):
        frame_idx = self._get_frame(page_id, assume_zero=True)
        self.frames[frame_idx][...] = data
        self._mark_frame_changed(page_id[0], frame_idx)
        self._enqueue_write(page_id, self.frames[frame_idx])
        self.frame_dirty[frame_idx] = False

    def write_kv_slice(
        self,
        page_id: Tuple[int, int, int],
        page_offset: int,
        k_slice: np.ndarray,
        v_slice: np.ndarray,
        assume_zero: bool,
    ):
        frame_idx = self._get_frame(page_id, assume_zero=assume_zero)
        end = page_offset + k_slice.shape[0]
        frame = self.frames[frame_idx]
        frame[0, page_offset:end] = k_slice
        frame[1, page_offset:end] = v_slice
        self.frame_dirty[frame_idx] = True
        self._mark_frame_changed(page_id[0], frame_idx)

    def write_kv_slice_direct(
        self,
        page_id: Tuple[int, int, int],
        page_offset: int,
        k_slice: np.ndarray,
        v_slice: np.ndarray,
    ):
        k_np = np.ascontiguousarray(k_slice, dtype=self.dtype)
        v_np = np.ascontiguousarray(v_slice, dtype=self.dtype)
        stride = self.head_dim * self.dtype.itemsize
        base = self._page_offset(page_id)
        k_offset = base + (0 * self.page_size + page_offset) * stride
        v_offset = base + (1 * self.page_size + page_offset) * stride
        self._enqueue_raw_write(k_offset, np.array(k_np, copy=True))
        self._enqueue_raw_write(v_offset, np.array(v_np, copy=True))
        self._invalidate_page(page_id)

    def flush(self):
        for page_id, frame_idx in list(self.page_table.items()):
            if self.frame_dirty[frame_idx]:
                self._sync_host_frame_from_resident(frame_idx)
                self._enqueue_write(page_id, self.frames[frame_idx])
                self.frame_dirty[frame_idx] = False
        if self._write_queue is not None:
            self._write_queue.join()

    def reset(self):
        self.flush()
        self.page_table.clear()
        for layer_lru in self.lru:
            layer_lru.clear()
        for hot in self.selected_hot:
            hot.clear()
        self.free_list = []
        base = 0
        for count in self._layer_frame_counts:
            self.free_list.append(list(range(base, base + count)))
            base += count
        self.frame_dirty = [False] * self.num_frames
        self.frame_page_id = [None] * self.num_frames
        self.frame_mx_cache = [None] * self.num_frames
        self.frame_version = [0] * self.num_frames
        self.frame_host_versions = [0] * self.num_frames
        self.metal_kv_storage = MetalKvFrameStorage(
            self.num_frames,
            self.page_size,
            self.mx_dtype,
        )
        self.resident_frames_mx = self.metal_kv_storage.buffers
        self.resident_frame_versions = self.metal_kv_storage.versions
        self._resident_storage_enabled = False
        self.layer_frames_mx_cache = [None for _ in range(self.num_layers)]
        self.layer_frames_mx_version_cache = [None for _ in range(self.num_layers)]
        self.hot_frames_mx_cache = [None for _ in range(self.num_layers)]
        self.hot_frame_versions = [None for _ in range(self.num_layers)]
        self.hot_frame_keys = [None for _ in range(self.num_layers)]
        self.hot_frame_lru = [OrderedDict() for _ in range(self.num_layers)]
        self.hot_frame_free = [None for _ in range(self.num_layers)]

    def close(self):
        self.flush()
        if self._write_queue is not None:
            self._write_queue.put(None)
            self._write_queue.join()
        os.close(self.fd)

    def pop_lru_stats(self) -> Tuple[int, int]:
        hits = self.hits
        misses = self.misses
        self.hits = 0
        self.misses = 0
        return hits, misses

    def pop_selected_page_stats(self) -> Tuple[int, int]:
        hits = self.selected_hits
        misses = self.selected_misses
        self.selected_hits = 0
        self.selected_misses = 0
        self.selected_requests = 0
        return hits, misses


class DiskOffloadKvCache:
    """
    Disk-offloaded KV Cache using a user-space buffer pool.
    Stores head-sliced pages in a monolithic file and caches them in RAM.
    """
    def __init__(
        self,
        num_layers: int,
        num_heads: int, # This is actually num_kv_heads now
        head_dim: int,
        max_seq_len: int,
        page_size: int,
        dtype=np.float32,
        mx_dtype=mx.float16,
        cache_dir: str = "./kv_cache_tmp",
        name: str = "kv_cache",
        async_disk_write: bool = False,
        buffer_pool_pages: Optional[int] = None,
        disable_os_cache: bool = True,
        release_active_buffer_on_prefill: bool = False,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads # Stores num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = dtype
        self.mx_dtype = mx_dtype
        self.name = name
        self.cache_dir = cache_dir
        self.async_disk_write = async_disk_write
        self.release_active_buffer_on_prefill = release_active_buffer_on_prefill
        self.write_through = False

        # Capacity in number of pages/blocks
        self.capacity = (max_seq_len + page_size - 1) // page_size
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        self.backing_file = os.path.join(cache_dir, f"{name}_pool.bin")
        
        # Determine shape: (num_layers, capacity, 2, page_size, num_heads, head_dim)
        # 2 represents Key and Value
        self.shape = (num_layers, self.capacity, 2, page_size, num_heads, head_dim)

        if buffer_pool_pages is None:
            buffer_pool_pages = min(self.capacity, 64)
        buffer_pool_pages = max(1, min(buffer_pool_pages, self.capacity))
        self.buffer_pool_pages = buffer_pool_pages
        num_frames = num_layers * buffer_pool_pages * num_heads
        self.buffer_pool = BufferPool(
            backing_file=self.backing_file,
            num_layers=num_layers,
            capacity=self.capacity,
            num_heads=num_heads,
            page_size=page_size,
            head_dim=head_dim,
            dtype=self.dtype,
            mx_dtype=self.mx_dtype,
            num_frames=num_frames,
            async_write=self.async_disk_write,
            disable_os_cache=disable_os_cache,
            on_evict=self._on_buffer_pool_evict,
        )
        
        # Track free blocks (simple stack allocator)
        self.free_slots = set(range(self.capacity))
        
        # Active page indices (ordered list of pages used by the sequence)
        self.active_indices: List[int] = []
        
        # Active buffers in MX: layer_idx -> mx.array
        self.active_buffers_mx: Dict[int, mx.array] = {}
        self.decode_rolling_buffers_mx: List[OrderedDict] = [
            OrderedDict() for _ in range(num_layers)
        ]
        self.decode_rolling_dirty_pages: List[Set[int]] = [
            set() for _ in range(num_layers)
        ]
        self.decode_rolling_flushed_pages: List[Set[int]] = [
            set() for _ in range(num_layers)
        ]
        self.decode_rolling_flush_batches = 0
        self._last_page_written_layers: Set[int] = set()
        self.page_cached_all = np.zeros((num_layers, self.capacity), dtype=np.bool_)
        self.page_arena_mx_cache: Dict[Tuple[int, str], Tuple[int, mx.array]] = {}
        self._metal_resident_frame_cache: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...], int], Tuple[mx.array, mx.array]] = {}
        
        self.seq_len = 0

    @property
    def last_page_len(self) -> int:
        if self.seq_len == 0:
            return 0
        rem = self.seq_len % self.page_size
        return self.page_size if rem == 0 else rem

    def alloc_block(self) -> int:
        if not self.free_slots:
            raise RuntimeError(f"KV Cache {self.name} full (capacity: {self.capacity} blocks)")
        idx = self.free_slots.pop()
        return idx
    
    def free_block(self, idx: int):
        if idx not in self.free_slots:
            self.free_slots.add(idx)

    def append_seq(self, seq_len: int) -> int:
        """Reserve space for tokens and return number of new pages allocated."""
        if seq_len <= 0:
            return 0
        resident_decode_append = (
            seq_len == 1
            and os.environ.get("ALAYAJET_QUEST_RESIDENT_FRAME_WRITE", "0").lower()
            in ("1", "true", "yes", "on")
        )
            
        appended_page_count = 0
        for _ in range(seq_len):
            # If current last page is full (or no pages yet), allocate a new one
            if self.seq_len % self.page_size == 0:
                # Flush previous active buffers if they exist
                if self.active_indices:
                    prev_page_idx = self.active_indices[-1]
                    if resident_decode_append:
                        for l, buffer_mx in list(self.active_buffers_mx.items()):
                            self.remember_decode_rolling_buffer(
                                l,
                                prev_page_idx,
                                buffer_mx,
                                dirty=False,
                            )
                        self.flush_decode_rolling_full_pages()
                    else:
                        self.flush_active_buffers(prev_page_idx)
                
                new_idx = self.alloc_block()
                self.active_indices.append(new_idx)
                appended_page_count += 1
                self._last_page_written_layers.clear()
                if self.release_active_buffer_on_prefill:
                    self.active_buffers_mx.clear()
                else:
                    # Initialize active buffers for all layers (MX only).
                    for l in range(self.num_layers):
                        # Shape: (2, page_size, num_heads, head_dim)
                        self.active_buffers_mx[l] = mx.zeros(
                            (2, self.page_size, self.num_heads, self.head_dim),
                            dtype=self.mx_dtype
                        )
                        if resident_decode_append:
                            self.remember_decode_rolling_buffer(
                                l,
                                new_idx,
                                self.active_buffers_mx[l],
                                dirty=False,
                            )
                    if resident_decode_append:
                        self.trim_decode_rolling_buffers()
            self.seq_len += 1
        return appended_page_count

    def remember_decode_rolling_buffer(
        self,
        layer_idx: int,
        page_idx: int,
        buffer_mx: mx.array,
        *,
        dirty: bool = True,
    ):
        layer_idx = int(layer_idx)
        buffers = self.decode_rolling_buffers_mx[int(layer_idx)]
        page_idx = int(page_idx)
        buffers[page_idx] = buffer_mx
        buffers.move_to_end(page_idx, last=True)
        if dirty:
            self.decode_rolling_dirty_pages[layer_idx].add(page_idx)
            self.decode_rolling_flushed_pages[layer_idx].discard(page_idx)

    def flush_decode_rolling_full_pages(self):
        for layer_idx, buffers in enumerate(self.decode_rolling_buffers_mx):
            dirty = self.decode_rolling_dirty_pages[layer_idx]
            pages_to_flush = [
                int(page_idx)
                for page_idx in buffers.keys()
                if int(page_idx) in dirty
            ][:2]
            if len(pages_to_flush) < 2:
                continue
            flushed = self.decode_rolling_flushed_pages[layer_idx]
            for page_idx in pages_to_flush:
                self._write_decode_rolling_buffer_page(
                    layer_idx,
                    page_idx,
                    buffers[page_idx],
                )
                dirty.discard(page_idx)
                flushed.add(page_idx)
            self.decode_rolling_flush_batches += 1

    def trim_decode_rolling_buffers(self):
        for layer_idx, buffers in enumerate(self.decode_rolling_buffers_mx):
            dirty = self.decode_rolling_dirty_pages[layer_idx]
            flushed = self.decode_rolling_flushed_pages[layer_idx]
            while len(buffers) > 2:
                page_idx, buffer_mx = buffers.popitem(last=False)
                if page_idx in dirty:
                    self._write_decode_rolling_buffer_page(
                        layer_idx,
                        page_idx,
                        buffer_mx,
                    )
                    dirty.discard(page_idx)
                flushed.discard(page_idx)
    
    def _write_active_buffer_page(self, layer_idx: int, page_idx: int, buffer_mx: mx.array):
        buffer_np = safe_to_numpy(buffer_mx, dtype=self.dtype)
        for kv_head in range(self.num_heads):
            page_id = (layer_idx, page_idx, kv_head)
            head_slice = np.ascontiguousarray(buffer_np[:, :, kv_head, :])
            if self.write_through:
                self.buffer_pool.write_page_direct(page_id, head_slice)
            else:
                self.buffer_pool.write_page(page_id, head_slice)
        if not self.write_through:
            self.page_cached_all[layer_idx, page_idx] = True
            for key in list(self.page_arena_mx_cache):
                if key[0] == layer_idx:
                    del self.page_arena_mx_cache[key]

    def _write_decode_rolling_buffer_page(
        self,
        layer_idx: int,
        page_idx: int,
        buffer_mx: mx.array,
    ):
        buffer_np = safe_to_numpy(buffer_mx, dtype=self.dtype)
        for kv_head in range(self.num_heads):
            page_id = (layer_idx, page_idx, kv_head)
            head_slice = np.ascontiguousarray(buffer_np[:, :, kv_head, :])
            self.buffer_pool.write_page_direct(page_id, head_slice)
        self.page_cached_all[layer_idx, page_idx] = False
        for key in list(self.page_arena_mx_cache):
            if key[0] == layer_idx:
                del self.page_arena_mx_cache[key]

    def _write_active_buffer_page_async(self, layer_idx: int, page_idx: int, buffer_mx: mx.array):
        buffer_np = safe_to_numpy(buffer_mx, dtype=self.dtype)
        for kv_head in range(self.num_heads):
            page_id = (layer_idx, page_idx, kv_head)
            head_slice = np.ascontiguousarray(buffer_np[:, :, kv_head, :])
            if self.write_through:
                self.buffer_pool.write_page_direct(page_id, head_slice)
            else:
                self.buffer_pool.write_page_async(page_id, head_slice)
        if not self.write_through:
            self.page_cached_all[layer_idx, page_idx] = True
            for key in list(self.page_arena_mx_cache):
                if key[0] == layer_idx:
                    del self.page_arena_mx_cache[key]

    def flush_active_buffers(self, page_idx: int):
        """Writes active buffers to buffer pool."""
        for l, buffer_mx in list(self.active_buffers_mx.items()):
            self._write_active_buffer_page(l, page_idx, buffer_mx)

    def get_active_buffer(self, layer_idx: int) -> Optional[np.ndarray]:
        return None

    def get_active_buffer_mx(self, layer_idx: int) -> Optional[mx.array]:
        buffer_mx = self.active_buffers_mx.get(layer_idx)
        if buffer_mx is not None:
            return buffer_mx
        if not self.active_indices:
            return None
        page_idx = self.active_indices[-1]
        if layer_idx in self._last_page_written_layers:
            buffer_mx = self.load_page_mx(layer_idx, page_idx)
        else:
            buffer_mx = mx.zeros(
                (2, self.page_size, self.num_heads, self.head_dim),
                dtype=self.mx_dtype,
            )
        self.active_buffers_mx[layer_idx] = buffer_mx
        return buffer_mx

    def offload_active_buffer(self, layer_idx: int):
        if not self.active_indices:
            return
        buffer_mx = self.active_buffers_mx.pop(layer_idx, None)
        if buffer_mx is None:
            return
        page_idx = self.active_indices[-1]
        self._write_active_buffer_page(layer_idx, page_idx, buffer_mx)
        self._last_page_written_layers.add(layer_idx)

    def offload_active_buffer_async(self, layer_idx: int):
        if not self.active_indices:
            return
        buffer_mx = self.active_buffers_mx.pop(layer_idx, None)
        if buffer_mx is None:
            return
        page_idx = self.active_indices[-1]
        self._write_active_buffer_page_async(layer_idx, page_idx, buffer_mx)
        self._last_page_written_layers.add(layer_idx)

    def load_pages(self, layer_idx: int, page_indices: List[int]) -> mx.array:
        """
        Load multiple pages from buffer pool.
        Returns mx.array of shape (num_pages, 2, page_size, num_heads, head_dim)
        """
        if not page_indices:
            return mx.array([])
        pages = []
        for page_idx in page_indices:
            heads = []
            for kv_head in range(self.num_heads):
                page_id = (layer_idx, page_idx, kv_head)
                head_page = self.buffer_pool.get_page(page_id)
                heads.append(head_page)
            page = np.stack(heads, axis=2)
            pages.append(page)
        pages_np = np.stack(pages, axis=0)
        return mx.array(pages_np)

    def load_pages_mx(self, layer_idx: int, page_indices: List[int], dtype=None) -> mx.array:
        """
        Load multiple pages using the per-frame MLX cache.
        Returns shape (num_pages, 2, page_size, num_heads, head_dim).
        """
        if not page_indices:
            return mx.array([])
        pages = [self.load_page_mx(layer_idx, int(page_idx)) for page_idx in page_indices]
        pages_mx = mx.stack(pages, axis=0)
        target_dtype = dtype if dtype is not None else self.mx_dtype
        if pages_mx.dtype != target_dtype:
            pages_mx = pages_mx.astype(target_dtype)
        return pages_mx

    def get_layer_page_arena_mx(
        self,
        layer_idx: int,
        page_indices: List[int],
        *,
        dtype,
        layout: str = "head_major",
    ) -> Tuple[np.ndarray, mx.array, str]:
        """Return a layer page arena keyed by physical page id.

        The arena is materialized once per layer/page-count and reused across
        decode steps.  Selected-page decode then passes page offsets into this
        arena instead of building a selected compact K/V tensor every step.
        """
        if not page_indices:
            if layout == "head_major":
                empty = mx.zeros(
                    (self.num_heads, 0, 2, self.page_size, self.head_dim),
                    dtype=dtype,
                )
            else:
                empty = mx.zeros(
                    (0, 2, self.page_size, self.num_heads, self.head_dim),
                    dtype=dtype,
                )
            return np.zeros((0,), dtype=np.int64), empty, layout

        physical_pages = np.asarray(page_indices, dtype=np.int64)
        max_page = int(physical_pages.max())
        cache_key = (layer_idx, layout)
        cached = self.page_arena_mx_cache.get(cache_key)
        if cached is None or cached[0] < max_page + 1:
            page_count = max_page + 1
            pages = []
            for page_idx in range(page_count):
                pages.append(self.load_page_mx(layer_idx, page_idx))
            arena = mx.stack(pages, axis=0)
            if layout == "head_major":
                arena = arena.transpose(3, 0, 1, 2, 4)
            elif layout != "page_major":
                raise ValueError("layout must be 'page_major' or 'head_major'")
            self.page_arena_mx_cache[cache_key] = (page_count, arena)
        else:
            page_count, arena = cached

        target_dtype = dtype if dtype is not None else self.mx_dtype
        if arena.dtype != target_dtype:
            arena = arena.astype(target_dtype)
        return physical_pages.astype(np.int64, copy=False), arena, layout

    def load_page_mx(self, layer_idx: int, page_idx: int) -> mx.array:
        heads = []
        for kv_head in range(self.num_heads):
            page_id = (layer_idx, page_idx, kv_head)
            heads.append(self.buffer_pool.get_page_mx(page_id))
        page_mx = mx.stack(heads, axis=2)
        if page_mx.dtype != self.mx_dtype:
            page_mx = page_mx.astype(self.mx_dtype)
        self.page_cached_all[layer_idx, page_idx] = True
        return page_mx

    def read_page_mx_from_disk(
        self,
        layer_idx: int,
        page_idx: int,
        dtype=None,
    ) -> mx.array:
        """Read one clean history page directly from the backing file.

        This bypasses BufferPool frame allocation and LRU state. It is used by
        the resident selected-page miss path, where the destination cache is the
        fixed Metal resident slot arena rather than the old host BufferPool.
        """
        layer_idx = int(layer_idx)
        page_idx = int(page_idx)
        if page_idx < 0 or page_idx >= self.capacity:
            raise RuntimeError(
                f"page_idx {page_idx} outside disk KV capacity {self.capacity}"
            )
        page_bytes = self.buffer_pool.page_bytes
        offset = (
            (layer_idx * self.capacity + page_idx)
            * self.num_heads
            * page_bytes
        )
        total_bytes = self.num_heads * page_bytes
        data = os.pread(self.buffer_pool.fd, total_bytes, offset)
        if len(data) != total_bytes:
            raise RuntimeError(
                f"Short read for page {page_idx} (layer {layer_idx}): "
                f"{len(data)} != {total_bytes}"
            )
        self.buffer_pool.read_bytes += total_bytes
        page_np = np.frombuffer(data, dtype=self.dtype).reshape(
            self.num_heads,
            2,
            self.page_size,
            self.head_dim,
        )
        page_mx = mx.array(page_np.transpose(1, 2, 0, 3))
        target_dtype = dtype if dtype is not None else self.mx_dtype
        if page_mx.dtype != target_dtype:
            page_mx = page_mx.astype(target_dtype)
        return page_mx

    def _read_pages_all_heads_into_cache(
        self,
        layer_idx: int,
        run_start: int,
        run_end: int,
    ) -> None:
        page_count = run_end - run_start + 1
        if page_count <= 0:
            return
        pages_np = self.buffer_pool._read_pages_all_heads(
            layer_idx,
            run_start,
            page_count,
        )
        for i in range(page_count):
            page_idx = run_start + i
            for kv_head in range(self.num_heads):
                page_id = (layer_idx, page_idx, kv_head)
                if self.buffer_pool.has_page(page_id):
                    continue
                self.buffer_pool.insert_page_cache(page_id, pages_np[i, kv_head])
            self.page_cached_all[layer_idx, page_idx] = True

    def load_kv_slices(
        self,
        layer_idx: int,
        physical_indices: np.ndarray,
        kv_head_indices: np.ndarray,
        timing_hook=None
    ) -> np.ndarray:
        """
        Load KV slices for each (page_idx, kv_head_idx).
        Returns numpy array of shape (H_q, k, 2, page_size, head_dim).
        """
        num_heads, k = physical_indices.shape
        if physical_indices.size == 0:
            return np.zeros(
                (num_heads, 0, 2, self.page_size, self.head_dim),
                dtype=self.dtype,
            )
        pages_np = np.empty(
            (num_heads, k, 2, self.page_size, self.head_dim),
            dtype=self.dtype,
        )
        for h in range(num_heads):
            kv_head = int(kv_head_indices[h, 0])
            for j in range(k):
                page_idx = int(physical_indices[h, j])
                page_id = (layer_idx, page_idx, kv_head)
                pages_np[h, j] = self.buffer_pool.get_page(page_id)
        return pages_np

    def load_kv_slices_mx(
        self,
        layer_idx: int,
        physical_indices: np.ndarray,
        kv_head_indices: np.ndarray,
        dtype,
    ) -> Tuple[mx.array, mx.array]:
        num_heads, k = physical_indices.shape
        if physical_indices.size == 0:
            empty = mx.zeros((num_heads, 0, self.head_dim), dtype=dtype)
            return empty, empty
        physical_indices = np.asarray(physical_indices, dtype=np.int64)
        kv_head_indices = np.asarray(kv_head_indices, dtype=np.int64).reshape(num_heads)
        selected_page_ids = []
        for h in range(num_heads):
            kv_head = int(kv_head_indices[h])
            for page_idx in physical_indices[h]:
                selected_page_ids.append((layer_idx, int(page_idx), kv_head))
        self.buffer_pool.mark_selected_pages(selected_page_ids)
        # Batch read contiguous pages and warm all KV heads for missing pages.
        union_pages_all = np.unique(physical_indices)
        if union_pages_all.size:
            cached_mask = self.page_cached_all[layer_idx, union_pages_all]
            if not cached_mask.all():
                missing_pages = union_pages_all[~cached_mask]
                missing_pages = np.sort(missing_pages)
                run_start = int(missing_pages[0])
                run_end = run_start
                for page_idx in missing_pages[1:]:
                    page_idx = int(page_idx)
                    if page_idx == run_end + 1:
                        run_end = page_idx
                        continue
                    self._read_pages_all_heads_into_cache(layer_idx, run_start, run_end)
                    run_start = run_end = page_idx
                self._read_pages_all_heads_into_cache(layer_idx, run_start, run_end)
        pages = [None] * (num_heads * k)
        for kv_head in np.unique(kv_head_indices):
            head_indices = np.nonzero(kv_head_indices == kv_head)[0]
            if head_indices.size == 0:
                continue
            pages_for_heads = physical_indices[head_indices]
            union_pages, inverse = np.unique(pages_for_heads, return_inverse=True)
            union_pages_mx = []
            for page_idx in union_pages:
                page_id = (layer_idx, int(page_idx), int(kv_head))
                union_pages_mx.append(self.buffer_pool.get_page_mx(page_id))
            pos = inverse.reshape(pages_for_heads.shape)
            for local_idx, head_idx in enumerate(head_indices):
                head_pos = pos[local_idx]
                base = int(head_idx) * k
                for j, union_idx in enumerate(head_pos):
                    pages[base + j] = union_pages_mx[int(union_idx)]
        pages_mx = mx.stack(pages, axis=0)
        if pages_mx.dtype != dtype:
            pages_mx = pages_mx.astype(dtype)
        pages_mx = pages_mx.reshape(
            num_heads,
            k,
            2,
            self.page_size,
            self.head_dim,
        )
        k_disk = pages_mx[:, :, 0].reshape(num_heads, -1, self.head_dim)
        v_disk = pages_mx[:, :, 1].reshape(num_heads, -1, self.head_dim)
        return k_disk, v_disk

    def mark_selected_kv_pages(
        self,
        layer_idx: int,
        physical_indices: np.ndarray,
        kv_head_indices: np.ndarray,
    ):
        physical_indices = np.asarray(physical_indices, dtype=np.int64)
        if physical_indices.size == 0:
            return
        num_heads = physical_indices.shape[0]
        kv_head_indices = np.asarray(kv_head_indices, dtype=np.int64).reshape(num_heads)
        selected_page_ids = []
        for h in range(num_heads):
            kv_head = int(kv_head_indices[h])
            for page_idx in physical_indices[h]:
                selected_page_ids.append((layer_idx, int(page_idx), kv_head))
        self.buffer_pool.mark_selected_pages(selected_page_ids)

    def load_selected_frame_indices_mx(
        self,
        layer_idx: int,
        physical_indices: np.ndarray,
        kv_head_indices: np.ndarray,
        dtype,
    ) -> Tuple[mx.array, mx.array]:
        """Load selected pages as compact resident frames plus per-head frame ids.

        Returns:
          selected_frames: ``(Hq, selected_count)`` int32 offsets into ``frames``.
          frames: ``(num_compact_frames, 2, page_size, head_dim)``.
        """
        physical_indices = np.asarray(physical_indices, dtype=np.int64)
        num_heads, selected_count = physical_indices.shape
        if physical_indices.size == 0:
            return (
                mx.zeros((num_heads, 0), dtype=mx.int32),
                mx.zeros((0, 2, self.page_size, self.head_dim), dtype=dtype),
            )
        kv_head_indices = np.asarray(kv_head_indices, dtype=np.int64).reshape(num_heads)
        selected_frame_offsets = np.zeros((num_heads, selected_count), dtype=np.int32)
        compact_frame_indices = []
        compact_lookup = {}
        for h in range(num_heads):
            kv_head = int(kv_head_indices[h])
            for j, page_idx in enumerate(physical_indices[h]):
                page_id = (layer_idx, int(page_idx), kv_head)
                frame_idx = self.buffer_pool.get_frame_index(page_id)
                compact_idx = compact_lookup.get(frame_idx)
                if compact_idx is None:
                    compact_idx = len(compact_frame_indices)
                    compact_lookup[frame_idx] = compact_idx
                    compact_frame_indices.append(frame_idx)
                selected_frame_offsets[h, j] = compact_idx
        frames = self.buffer_pool.get_frames_mx(compact_frame_indices, dtype=dtype)
        return mx.array(selected_frame_offsets), frames

    def load_selected_direct_frame_indices_mx(
        self,
        layer_idx: int,
        physical_indices: np.ndarray,
        kv_head_indices: np.ndarray,
        dtype,
    ) -> Tuple[mx.array, List[mx.array]]:
        """Return selected local frame ids plus unstacked frame buffers."""
        physical_indices = np.asarray(physical_indices, dtype=np.int64)
        num_heads, selected_count = physical_indices.shape
        if physical_indices.size == 0:
            return mx.zeros((num_heads, 0), dtype=mx.int32), []
        kv_head_indices = np.asarray(kv_head_indices, dtype=np.int64).reshape(num_heads)
        selected_frame_offsets = np.zeros((num_heads, selected_count), dtype=np.int32)
        compact_frame_indices = []
        compact_lookup = {}
        for h in range(num_heads):
            kv_head = int(kv_head_indices[h])
            for j, page_idx in enumerate(physical_indices[h]):
                page_id = (layer_idx, int(page_idx), kv_head)
                frame_idx = self.buffer_pool.get_frame_index(page_id)
                compact_idx = compact_lookup.get(frame_idx)
                if compact_idx is None:
                    compact_idx = len(compact_frame_indices)
                    compact_lookup[frame_idx] = compact_idx
                    compact_frame_indices.append(frame_idx)
                selected_frame_offsets[h, j] = compact_idx
        frames = self.buffer_pool.get_frame_buffers_mx(compact_frame_indices, dtype=dtype)
        return mx.array(selected_frame_offsets), frames

    def load_selected_layer_frame_indices_mx(
        self,
        layer_idx: int,
        physical_indices: np.ndarray,
        kv_head_indices: np.ndarray,
        dtype,
    ) -> Tuple[mx.array, mx.array]:
        """Return per-head local frame ids plus a layer-resident frame arena."""
        physical_indices = np.asarray(physical_indices, dtype=np.int64)
        num_heads, selected_count = physical_indices.shape
        if physical_indices.size == 0:
            return (
                mx.zeros((num_heads, 0), dtype=mx.int32),
                mx.zeros((0, 2, self.page_size, self.head_dim), dtype=dtype),
            )
        kv_head_indices = np.asarray(kv_head_indices, dtype=np.int64).reshape(num_heads)
        selected_local_frames = np.zeros((num_heads, selected_count), dtype=np.int32)
        required_frame_indices = []
        for h in range(num_heads):
            kv_head = int(kv_head_indices[h])
            for j, page_idx in enumerate(physical_indices[h]):
                page_id = (layer_idx, int(page_idx), kv_head)
                frame_idx = self.buffer_pool.get_frame_index(page_id)
                required_frame_indices.append(frame_idx)
                selected_local_frames[h, j] = self.buffer_pool.layer_local_frame_index(
                    layer_idx,
                    frame_idx,
                )
        frames = self.buffer_pool.get_layer_frames_mx(
            layer_idx,
            dtype=dtype,
            required_frame_indices=required_frame_indices,
        )
        return mx.array(selected_local_frames), frames

    def load_selected_hot_frame_indices_mx(
        self,
        layer_idx: int,
        physical_indices: np.ndarray,
        kv_head_indices: np.ndarray,
        dtype,
    ) -> Tuple[mx.array, mx.array]:
        """Return per-head hot-arena slot ids plus a persistent hot frame arena."""
        physical_indices = np.asarray(physical_indices, dtype=np.int64)
        num_heads, selected_count = physical_indices.shape
        if physical_indices.size == 0:
            return (
                mx.zeros((num_heads, 0), dtype=mx.int32),
                mx.zeros((0, 2, self.page_size, self.head_dim), dtype=dtype),
            )
        kv_head_indices = np.asarray(kv_head_indices, dtype=np.int64).reshape(num_heads)
        page_ids = []
        for h in range(num_heads):
            kv_head = int(kv_head_indices[h])
            for page_idx in physical_indices[h]:
                page_ids.append((layer_idx, int(page_idx), kv_head))
        unique_page_ids = list(dict.fromkeys(page_ids))
        frame_indices_by_page = {
            page_id: self.buffer_pool.get_frame_index(page_id)
            for page_id in unique_page_ids
        }
        unique_frame_indices = [frame_indices_by_page[page_id] for page_id in unique_page_ids]
        unique_slots, frames = self.buffer_pool.get_hot_selected_frames_mx(
            layer_idx,
            unique_page_ids,
            unique_frame_indices,
            dtype=dtype,
        )
        slot_by_page = {
            page_id: int(unique_slots[i])
            for i, page_id in enumerate(unique_page_ids)
        }
        selected_slots = np.empty((num_heads, selected_count), dtype=np.int32)
        for linear_idx, page_id in enumerate(page_ids):
            h = linear_idx // selected_count
            j = linear_idx - h * selected_count
            selected_slots[h, j] = slot_by_page[page_id]
        return mx.array(selected_slots), frames

    def load_selected_resident_frame_indices_mx(
        self,
        layer_idx: int,
        physical_indices: np.ndarray,
        kv_head_indices: np.ndarray,
        dtype,
    ) -> Tuple[mx.array, List[mx.array]]:
        """Return stable layer-local resident frame ids plus resident buffers."""
        physical_indices = np.asarray(physical_indices, dtype=np.int64)
        num_heads, selected_count = physical_indices.shape
        if physical_indices.size == 0:
            return (
                mx.zeros((num_heads, 0), dtype=mx.int32),
                [],
            )
        kv_head_indices = np.asarray(kv_head_indices, dtype=np.int64).reshape(num_heads)
        selected_frames = np.empty((num_heads, selected_count), dtype=np.int32)
        required_frame_indices = []
        for h in range(num_heads):
            kv_head = int(kv_head_indices[h])
            for j, page_idx in enumerate(physical_indices[h]):
                page_id = (layer_idx, int(page_idx), kv_head)
                frame_idx = self.buffer_pool.get_frame_index(page_id)
                required_frame_indices.append(frame_idx)
                selected_frames[h, j] = self.buffer_pool.layer_local_frame_index(
                    layer_idx,
                    frame_idx,
                )
        frames = self.buffer_pool.get_resident_layer_slot_buffers_mx(
            layer_idx,
            list(dict.fromkeys(required_frame_indices)),
            dtype=dtype,
        )
        return mx.array(selected_frames), frames

    def load_selected_resident_frame_indices_with_last_mx(
        self,
        layer_idx: int,
        physical_indices: np.ndarray,
        kv_head_indices: np.ndarray,
        last_page_idx: int,
        dtype,
    ) -> Tuple[mx.array, List[mx.array]]:
        """Return resident frame ids with the active last page appended."""
        physical_indices = np.asarray(physical_indices, dtype=np.int64)
        num_heads, selected_count = physical_indices.shape
        kv_head_indices = np.asarray(kv_head_indices, dtype=np.int64).reshape(num_heads)
        selected_frames = np.empty((num_heads, selected_count + 1), dtype=np.int32)
        required_frame_indices = []
        for h in range(num_heads):
            kv_head = int(kv_head_indices[h])
            for j, page_idx in enumerate(physical_indices[h]):
                page_id = (layer_idx, int(page_idx), kv_head)
                frame_idx = self.buffer_pool.get_frame_index(page_id)
                required_frame_indices.append(frame_idx)
                selected_frames[h, j] = self.buffer_pool.layer_local_frame_index(
                    layer_idx,
                    frame_idx,
                )
            last_page_id = (layer_idx, int(last_page_idx), kv_head)
            last_frame_idx = self.buffer_pool.get_frame_index(last_page_id)
            required_frame_indices.append(last_frame_idx)
            selected_frames[h, selected_count] = self.buffer_pool.layer_local_frame_index(
                layer_idx,
                last_frame_idx,
            )
        frames = self.buffer_pool.get_resident_layer_slot_buffers_mx(
            layer_idx,
            list(dict.fromkeys(required_frame_indices)),
            dtype=dtype,
        )
        return mx.array(selected_frames), frames

    def load_selected_resident_layer_frame_indices_with_last_mx(
        self,
        layer_idx: int,
        physical_indices: np.ndarray,
        kv_head_indices: np.ndarray,
        last_page_idx: int,
        dtype,
    ) -> Tuple[mx.array, mx.array]:
        physical_indices = np.asarray(physical_indices, dtype=np.int64)
        num_heads, selected_count = physical_indices.shape
        kv_head_indices = np.asarray(kv_head_indices, dtype=np.int64).reshape(num_heads)
        cache_key = (
            int(layer_idx),
            tuple(int(x) for x in physical_indices.reshape(-1).tolist()),
            tuple(int(x) for x in kv_head_indices.reshape(-1).tolist()),
            int(last_page_idx),
        )
        cached = self._metal_resident_frame_cache.get(cache_key)
        if cached is not None:
            return cached
        selected_frames = np.empty((num_heads, selected_count + 1), dtype=np.int32)
        required_frame_indices = []
        for h in range(num_heads):
            kv_head = int(kv_head_indices[h])
            for j, page_idx in enumerate(physical_indices[h]):
                page_id = (layer_idx, int(page_idx), kv_head)
                frame_idx = self.buffer_pool.get_frame_index(page_id)
                required_frame_indices.append(frame_idx)
                selected_frames[h, j] = self.buffer_pool.layer_local_frame_index(
                    layer_idx,
                    frame_idx,
                )
            last_page_id = (layer_idx, int(last_page_idx), kv_head)
            last_frame_idx = self.buffer_pool.get_frame_index(last_page_id)
            required_frame_indices.append(last_frame_idx)
            selected_frames[h, selected_count] = self.buffer_pool.layer_local_frame_index(
                layer_idx,
                last_frame_idx,
            )
        frames = self.buffer_pool.get_layer_frames_mx(
            layer_idx,
            dtype=dtype,
            required_frame_indices=list(dict.fromkeys(required_frame_indices)),
        )
        result = (mx.array(selected_frames), frames)
        for stale_key in list(self._metal_resident_frame_cache):
            if stale_key[0] == layer_idx:
                del self._metal_resident_frame_cache[stale_key]
        self._metal_resident_frame_cache[cache_key] = result
        return result

    def write_kv_slice(
        self,
        layer_idx: int,
        page_idx: int,
        page_offset: int,
        k_np: np.ndarray,
        v_np: np.ndarray,
        assume_zero: bool,
        write_through: bool = False,
    ):
        for kv_head in range(self.num_heads):
            page_id = (layer_idx, page_idx, kv_head)
            k_slice = k_np[:, kv_head, :]
            v_slice = v_np[:, kv_head, :]
            if write_through:
                self.buffer_pool.write_kv_slice_direct(
                    page_id=page_id,
                    page_offset=page_offset,
                    k_slice=k_slice,
                    v_slice=v_slice,
                )
            else:
                self.buffer_pool.write_kv_slice(
                    page_id=page_id,
                    page_offset=page_offset,
                    k_slice=k_slice,
                    v_slice=v_slice,
                    assume_zero=assume_zero,
                )
        if write_through:
            self.page_cached_all[layer_idx, page_idx] = False
        else:
            self.page_cached_all[layer_idx, page_idx] = True
            for key in list(self.page_arena_mx_cache):
                if key[0] == layer_idx:
                    del self.page_arena_mx_cache[key]

    def write_kv_slice_mx_resident(
        self,
        layer_idx: int,
        page_idx: int,
        page_offset: int,
        k_mx: mx.array,
        v_mx: mx.array,
        assume_zero: bool,
    ):
        page_ids = [
            (layer_idx, page_idx, kv_head)
            for kv_head in range(self.num_heads)
        ]
        self.buffer_pool.write_kv_slices_mx_resident_batched(
            page_ids=page_ids,
            page_offset=page_offset,
            k_mx=k_mx,
            v_mx=v_mx,
            assume_zero=assume_zero,
        )
        self.page_cached_all[layer_idx, page_idx] = True
        for key in list(self.page_arena_mx_cache):
            if key[0] == layer_idx:
                del self.page_arena_mx_cache[key]

    def release(self):
        self.buffer_pool.reset()
        self.seq_len = 0
        # Reset free slots
        self.free_slots = set(range(self.capacity))
        self.active_indices.clear()
        self.active_buffers_mx.clear()
        for buffers in self.decode_rolling_buffers_mx:
            buffers.clear()
        for dirty in self.decode_rolling_dirty_pages:
            dirty.clear()
        for flushed in self.decode_rolling_flushed_pages:
            flushed.clear()
        self.decode_rolling_flush_batches = 0
        self._last_page_written_layers.clear()
        self.page_cached_all.fill(False)
        self.page_arena_mx_cache.clear()
        self._metal_resident_frame_cache.clear()

    def _on_buffer_pool_evict(self, page_id: Tuple[int, int, int]):
        layer_idx, page_idx, _ = page_id
        if 0 <= layer_idx < self.page_cached_all.shape[0] and 0 <= page_idx < self.page_cached_all.shape[1]:
            self.page_cached_all[layer_idx, page_idx] = False
            for key in list(self.page_arena_mx_cache):
                if key[0] == layer_idx:
                    del self.page_arena_mx_cache[key]

    def pop_lru_stats(self) -> Tuple[int, int]:
        return self.buffer_pool.pop_lru_stats()

    def pop_selected_page_stats(self) -> Tuple[int, int]:
        return self.buffer_pool.pop_selected_page_stats()


class QuestController:
    """
    Manages the KV Cache (Disk Offloaded) and Metadata Cache (RAM) for Quest.
    """
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int,
        page_budget: int,
        max_seq_len: int,
        num_kv_heads: Optional[int] = None, # Added parameter
        dtype=mx.float16,
        cache_dir: str = "./kv_cache_tmp",
        async_disk_write: bool = False,
        buffer_pool_pages: Optional[int] = None,
        disable_os_cache: bool = True,
        release_active_buffer_on_prefill: bool = False,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self._page_budget = page_budget
        self.dtype = dtype
        self.disable_os_cache = disable_os_cache
        if buffer_pool_pages is None:
            group_size = max(1, self.num_heads // self.num_kv_heads)
            buffer_pool_pages = page_budget * group_size * 4
        buffer_pool_pages = max(1, buffer_pool_pages)
        
        # Main KV Cache (Disk Backed)
        # Store in float16 when the model uses float16 to cut disk I/O in half.
        # Keep float32 for other dtypes to avoid precision loss.
        np_dtype = np.float16 if dtype == mx.float16 else np.float32
        
        self.kv_cache = DiskOffloadKvCache(
            num_layers=num_layers,
            num_heads=self.num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            page_size=page_size,
            dtype=np_dtype,
            mx_dtype=dtype,
            cache_dir=cache_dir,
            name="kv_cache",
            async_disk_write=async_disk_write,
            buffer_pool_pages=buffer_pool_pages,
            disable_os_cache=disable_os_cache,
            release_active_buffer_on_prefill=release_active_buffer_on_prefill,
        )
        max_kv_pages = (max_seq_len + page_size - 1) // page_size
        resident_slot_blocks = max(1, min(max_kv_pages, buffer_pool_pages))
        if max_kv_pages > 2:
            resident_slot_blocks = max(3, resident_slot_blocks)
        self.resident_frame_pool = ResidentKVSlotPool(
            num_layers=num_layers,
            num_heads=self.num_kv_heads,
            max_blocks=max_kv_pages,
            slot_blocks=resident_slot_blocks,
            page_size=page_size,
            head_dim=head_dim,
            mx_dtype=dtype,
        )
        
        # Metadata Cache (RAM)
        # Keeps min/max keys for each page.
        # Shape: (num_layers, capacity, 2, num_kv_heads, head_dim)
        
        # Keep metadata on host (numpy) to avoid device memory growth.
        self.metadata_pool = np.zeros(
            (num_layers, max_kv_pages, 2, self.num_kv_heads, head_dim),
            dtype=np.float32
        )
        
        self.inference_page_budget = page_budget
        
        # Current state
        self.kv_indices_with_last = []
        self.kv_indices_without_last = []
        self.prefill_io_timing = False
        self.prefill_write_through = False
        self._prefill_executor: Optional[ThreadPoolExecutor] = None
        self._prefill_buffers: Optional[List[np.ndarray]] = None
        self._prefill_buffer_spec: Optional[Tuple[int, int, int, int, np.dtype]] = None
        self._metadata_mx_cache: Dict[int, mx.array] = {}
        self._metadata_cache_len: Dict[int, int] = {}
        self._metadata_kminmax_cache: Dict[int, Tuple[mx.array, mx.array]] = {}
        self._metadata_kminmax_cache_len: Dict[int, int] = {}
        self._metal_sparse_cache: Dict[Tuple[int, Tuple[int, ...]], Tuple[mx.array, mx.array, mx.array]] = {}
        self._metal_selected_cache: Dict[Tuple[int, str, Tuple[int, ...]], mx.array] = {}
        self._metal_selected_frame_cache: Dict[Tuple, Tuple[mx.array, mx.array]] = {}
        self._metal_page_arena_index_cache: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], mx.array] = {}
        
    def set_page_budget(self, page_budget: int):
        self._page_budget = page_budget

    def set_prefill_write_through(self, enabled: bool):
        self.prefill_write_through = enabled
        self.kv_cache.write_through = enabled
        
    def prepare_metadata(self, seq_len: int):
        """
        Allocates pages in KV cache.
        """
        self.kv_cache.append_seq(seq_len)
        
    def update_metadata(self, layer_idx: int, page_idx: int, k_min: mx.array, k_max: mx.array):
        """
        Updates metadata for a specific page.
        """
        # k_min, k_max: (num_kv_heads, head_dim)
        k_min = safe_to_numpy(k_min, dtype=np.float32)
        k_max = safe_to_numpy(k_max, dtype=np.float32)
        self.metadata_pool[layer_idx, page_idx, 0] = k_min
        self.metadata_pool[layer_idx, page_idx, 1] = k_max

    def get_metadata_tensor(self, layer_idx: int, page_indices: List[int]) -> mx.array:
        """
        Returns metadata for selected pages as MLX tensor.
        """
        if not page_indices:
            return mx.array([])
        if page_indices is self.kv_indices_without_last:
            cached_len = self._metadata_cache_len.get(layer_idx)
            cached = self._metadata_mx_cache.get(layer_idx)
            if cached is not None and cached_len == len(page_indices):
                return cached

        data = self.metadata_pool[layer_idx, page_indices]
        # Transpose to (2, NumPages, H_kv, D) to match expected logic
        tensor = mx.array(data.transpose(1, 0, 2, 3))
        if page_indices is self.kv_indices_without_last:
            self._metadata_mx_cache[layer_idx] = tensor
            self._metadata_cache_len[layer_idx] = len(page_indices)
        return tensor

    def get_metadata_kminmax(
        self,
        layer_idx: int,
    ) -> Tuple[mx.array, mx.array]:
        """
        Returns K_min/K_max shaped for estimate, cached by page count.
        """
        pages_indices = self.kv_indices_without_last
        if not pages_indices:
            empty = mx.zeros((self.num_kv_heads, 0, self.head_dim), dtype=mx.float32)
            return empty, empty
        cached_len = self._metadata_kminmax_cache_len.get(layer_idx)
        cached = self._metadata_kminmax_cache.get(layer_idx)
        if cached is not None and cached_len == len(pages_indices):
            return cached
        metadata = self.get_metadata_tensor(layer_idx, pages_indices)
        k_min = metadata[0].transpose(1, 0, 2)
        k_max = metadata[1].transpose(1, 0, 2)
        self._metadata_kminmax_cache[layer_idx] = (k_min, k_max)
        self._metadata_kminmax_cache_len[layer_idx] = len(pages_indices)
        return k_min, k_max

    def begin_forward(self, seq_len: int):
        """
        Prepare indices for the forward pass.
        """
        cur_page_nums = len(self.kv_cache.active_indices)
        self.kv_indices_with_last = list(range(cur_page_nums))
        self.kv_indices_without_last = list(range(max(0, cur_page_nums - 1)))
        self.inference_page_budget = min(self._page_budget, cur_page_nums)
            
    def need_estimate(self) -> bool:
        if self.inference_page_budget is None:
            return False
        cur_page_nums = len(self.kv_cache.active_indices)
        return cur_page_nums > self.inference_page_budget
    
    def clean_states(self):
        self.kv_cache.release()
        self.close_prefill_resources()
        self._metadata_mx_cache.clear()
        self._metadata_cache_len.clear()
        self._metadata_kminmax_cache.clear()
        self._metadata_kminmax_cache_len.clear()
        self._metal_sparse_cache.clear()
        self._metal_selected_cache.clear()
        self._metal_selected_frame_cache.clear()
        self._metal_page_arena_index_cache.clear()

    def get_prefill_executor(self) -> ThreadPoolExecutor:
        if self._prefill_executor is None:
            self._prefill_executor = ThreadPoolExecutor(max_workers=1)
        return self._prefill_executor

    def get_prefill_buffers(
        self,
        chunk_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: np.dtype,
    ) -> List[np.ndarray]:
        spec = (chunk_pages, page_size, num_kv_heads, head_dim, np.dtype(dtype))
        if self._prefill_buffers is None or self._prefill_buffer_spec != spec:
            self._prefill_buffers = [
                np.empty(
                    (chunk_pages, 2, page_size, num_kv_heads, head_dim),
                    dtype=spec[4],
                ),
                np.empty(
                    (chunk_pages, 2, page_size, num_kv_heads, head_dim),
                    dtype=spec[4],
                ),
            ]
            self._prefill_buffer_spec = spec
        return self._prefill_buffers

    def close_prefill_resources(self):
        if self._prefill_executor is not None:
            self._prefill_executor.shutdown(wait=True)
            self._prefill_executor = None
        self._prefill_buffers = None
        self._prefill_buffer_spec = None
