import mlx.core as mx
import numpy as np
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict, Set, Callable

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
            np.empty(self.frame_shape, dtype=self.dtype)
            for _ in range(self.num_frames)
        ]
        self.frame_dirty = [False] * self.num_frames
        self.frame_page_id: List[Optional[Tuple[int, int, int]]] = [None] * self.num_frames
        self.frame_mx_cache: List[Optional[mx.array]] = [None] * self.num_frames

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

        self.read_bytes = 0
        self.write_bytes = 0
        self.hits = 0
        self.misses = 0

        self._write_queue: Optional[queue.Queue] = None
        self._write_thread: Optional[threading.Thread] = None
        self.async_write = async_write
        self._on_evict = on_evict
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
        self.free_list[layer_idx].append(frame_idx)
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
        page_id, frame_idx = layer_lru.popitem(last=False)
        if self._on_evict is not None:
            self._on_evict(page_id)
        if self.frame_dirty[frame_idx]:
            self._enqueue_write(page_id, self.frames[frame_idx])
            self.frame_dirty[frame_idx] = False
        del self.page_table[page_id]
        self.frame_page_id[frame_idx] = None
        self.frame_mx_cache[frame_idx] = None
        return frame_idx

    def _alloc_frame(self, layer_idx: int) -> int:
        layer_free = self.free_list[layer_idx]
        if layer_free:
            return layer_free.pop()
        return self._evict_frame(layer_idx)

    def _touch(self, page_id: Tuple[int, int, int]):
        layer_lru = self.lru[page_id[0]]
        if page_id in layer_lru:
            layer_lru.move_to_end(page_id, last=True)

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
        self.frame_mx_cache[frame_idx] = None
        return frame_idx

    def get_page(self, page_id: Tuple[int, int, int], assume_zero: bool = False) -> np.ndarray:
        frame_idx = self._get_frame(page_id, assume_zero)
        return self.frames[frame_idx]

    def get_page_mx(self, page_id: Tuple[int, int, int], assume_zero: bool = False) -> mx.array:
        frame_idx = self._get_frame(page_id, assume_zero)
        cached = self.frame_mx_cache[frame_idx]
        if cached is None:
            cached = mx.array(self.frames[frame_idx]).astype(self.mx_dtype)
            self.frame_mx_cache[frame_idx] = cached
        return cached

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
        self.frame_mx_cache[frame_idx] = None

    def write_page(self, page_id: Tuple[int, int, int], data: np.ndarray):
        frame_idx = self._get_frame(page_id, assume_zero=True)
        self.frames[frame_idx][...] = data
        self.frame_dirty[frame_idx] = True
        self.frame_mx_cache[frame_idx] = None

    def write_page_async(self, page_id: Tuple[int, int, int], data: np.ndarray):
        frame_idx = self._get_frame(page_id, assume_zero=True)
        self.frames[frame_idx][...] = data
        self.frame_mx_cache[frame_idx] = None
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
        self.frame_mx_cache[frame_idx] = None

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
                self._enqueue_write(page_id, self.frames[frame_idx])
                self.frame_dirty[frame_idx] = False
        if self._write_queue is not None:
            self._write_queue.join()

    def reset(self):
        self.flush()
        self.page_table.clear()
        for layer_lru in self.lru:
            layer_lru.clear()
        self.free_list = []
        base = 0
        for count in self._layer_frame_counts:
            self.free_list.append(list(range(base, base + count)))
            base += count
        self.frame_dirty = [False] * self.num_frames
        self.frame_page_id = [None] * self.num_frames
        self.frame_mx_cache = [None] * self.num_frames

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
        self._last_page_written_layers: Set[int] = set()
        self.page_cached_all = np.zeros((num_layers, self.capacity), dtype=np.bool_)
        
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
            
        appended_page_count = 0
        for _ in range(seq_len):
            # If current last page is full (or no pages yet), allocate a new one
            if self.seq_len % self.page_size == 0:
                # Flush previous active buffers if they exist
                if self.active_indices:
                    prev_page_idx = self.active_indices[-1]
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
            self.seq_len += 1
        return appended_page_count
    
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

    def release(self):
        self.buffer_pool.reset()
        self.seq_len = 0
        # Reset free slots
        self.free_slots = set(range(self.capacity))
        self.active_indices.clear()
        self.active_buffers_mx.clear()
        self._last_page_written_layers.clear()
        self.page_cached_all.fill(False)

    def _on_buffer_pool_evict(self, page_id: Tuple[int, int, int]):
        layer_idx, page_idx, _ = page_id
        if 0 <= layer_idx < self.page_cached_all.shape[0] and 0 <= page_idx < self.page_cached_all.shape[1]:
            self.page_cached_all[layer_idx, page_idx] = False

    def pop_lru_stats(self) -> Tuple[int, int]:
        return self.buffer_pool.pop_lru_stats()


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
        
        # Metadata Cache (RAM)
        # Keeps min/max keys for each page.
        # Shape: (num_layers, capacity, 2, num_kv_heads, head_dim)
        
        max_kv_pages = (max_seq_len + page_size - 1) // page_size
        
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
