import mlx.core as mx
import numpy as np
import os
import queue
import threading
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict

try:
    import fcntl
except Exception:  # pragma: no cover - optional on some platforms
    fcntl = None


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
        self.lru = OrderedDict()
        self.free_list = list(range(self.num_frames))

        self.read_bytes = 0
        self.write_bytes = 0
        self.hits = 0
        self.misses = 0

        self._write_queue: Optional[queue.Queue] = None
        self._write_thread: Optional[threading.Thread] = None
        self.async_write = async_write
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

    def _write_page(self, page_id: Tuple[int, int, int], frame: np.ndarray):
        offset = self._page_offset(page_id)
        written = os.pwrite(self.fd, frame, offset)
        if written != self.page_bytes:
            raise RuntimeError(
                f"Short write for page {page_id}: {written} != {self.page_bytes}"
            )
        self.write_bytes += self.page_bytes

    def _enqueue_write(self, page_id: Tuple[int, int, int], frame: np.ndarray):
        if self._write_queue is None:
            self._write_page(page_id, frame)
            return
        self._write_queue.put((page_id, np.array(frame, copy=True)))

    def _write_worker(self):
        if self._write_queue is None:
            return
        while True:
            item = self._write_queue.get()
            if item is None:
                self._write_queue.task_done()
                break
            page_id, frame = item
            self._write_page(page_id, frame)
            self._write_queue.task_done()

    def _evict_frame(self) -> int:
        page_id, frame_idx = self.lru.popitem(last=False)
        if self.frame_dirty[frame_idx]:
            self._enqueue_write(page_id, self.frames[frame_idx])
            self.frame_dirty[frame_idx] = False
        del self.page_table[page_id]
        self.frame_page_id[frame_idx] = None
        self.frame_mx_cache[frame_idx] = None
        return frame_idx

    def _alloc_frame(self) -> int:
        if self.free_list:
            return self.free_list.pop()
        return self._evict_frame()

    def _touch(self, page_id: Tuple[int, int, int]):
        if page_id in self.lru:
            self.lru.move_to_end(page_id, last=True)

    def _get_frame(self, page_id: Tuple[int, int, int], assume_zero: bool) -> int:
        frame_idx = self.page_table.get(page_id)
        if frame_idx is not None:
            self.hits += 1
            self._touch(page_id)
            return frame_idx
        self.misses += 1
        frame_idx = self._alloc_frame()
        self.page_table[page_id] = frame_idx
        self.frame_page_id[frame_idx] = page_id
        self.lru[page_id] = frame_idx
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

    def write_page(self, page_id: Tuple[int, int, int], data: np.ndarray):
        frame_idx = self._get_frame(page_id, assume_zero=True)
        self.frames[frame_idx][...] = data
        self.frame_dirty[frame_idx] = True
        self.frame_mx_cache[frame_idx] = None

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
        self.lru.clear()
        self.free_list = list(range(self.num_frames))
        self.frame_dirty = [False] * self.num_frames
        self.frame_page_id = [None] * self.num_frames
        self.frame_mx_cache = [None] * self.num_frames

    def close(self):
        self.flush()
        if self._write_queue is not None:
            self._write_queue.put(None)
            self._write_queue.join()
        os.close(self.fd)


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
        )
        
        # Track free blocks (simple stack allocator)
        self.free_slots = set(range(self.capacity))
        
        # Active page indices (ordered list of pages used by the sequence)
        self.active_indices: List[int] = []
        
        # Active buffers in MX: layer_idx -> mx.array
        self.active_buffers_mx: Dict[int, mx.array] = {}
        
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
                
                # Initialize active buffers for all layers (MX only).
                for l in range(self.num_layers):
                    # Shape: (2, page_size, num_heads, head_dim)
                    self.active_buffers_mx[l] = mx.zeros(
                        (2, self.page_size, self.num_heads, self.head_dim),
                        dtype=self.mx_dtype
                    )
            self.seq_len += 1
        return appended_page_count
    
    def flush_active_buffers(self, page_idx: int):
        """Writes active buffers to buffer pool."""
        for l, buffer_mx in self.active_buffers_mx.items():
            buffer_np = np.array(buffer_mx)
            if buffer_np.dtype != self.dtype:
                buffer_np = buffer_np.astype(self.dtype, copy=False)
            for kv_head in range(self.num_heads):
                page_id = (l, page_idx, kv_head)
                head_slice = np.ascontiguousarray(buffer_np[:, :, kv_head, :])
                self.buffer_pool.write_page(page_id, head_slice)

    def get_active_buffer(self, layer_idx: int) -> Optional[np.ndarray]:
        return None

    def get_active_buffer_mx(self, layer_idx: int) -> Optional[mx.array]:
        return self.active_buffers_mx.get(layer_idx)

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
        pages = []
        for h in range(num_heads):
            kv_head = int(kv_head_indices[h, 0])
            for j in range(k):
                page_idx = int(physical_indices[h, j])
                page_id = (layer_idx, page_idx, kv_head)
                pages.append(self.buffer_pool.get_page_mx(page_id))
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
    ):
        for kv_head in range(self.num_heads):
            page_id = (layer_idx, page_idx, kv_head)
            k_slice = k_np[:, kv_head, :]
            v_slice = v_np[:, kv_head, :]
            self.buffer_pool.write_kv_slice(
                page_id=page_id,
                page_offset=page_offset,
                k_slice=k_slice,
                v_slice=v_slice,
                assume_zero=assume_zero,
            )

    def release(self):
        self.buffer_pool.reset()
        self.seq_len = 0
        # Reset free slots
        self.free_slots = set(range(self.capacity))
        self.active_indices.clear()
        self.active_buffers_mx.clear()


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
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self._page_budget = page_budget
        self.dtype = dtype
        if buffer_pool_pages is None:
            group_size = max(1, self.num_heads // self.num_kv_heads)
            buffer_pool_pages = page_budget * group_size
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
        )
        
        # Metadata Cache (RAM)
        # Keeps min/max keys for each page.
        # Shape: (num_layers, capacity, 2, num_kv_heads, head_dim)
        
        max_kv_pages = (max_seq_len + page_size - 1) // page_size
        
        # Using numpy for mutable access (Metadata is small enough to stay in RAM)
        self.metadata_pool = np.zeros(
            (num_layers, max_kv_pages, 2, self.num_kv_heads, head_dim),
            dtype=np.float32
        )
        
        self.inference_page_budget = page_budget
        
        # Current state
        self.kv_indices_with_last = []
        self.kv_indices_without_last = []
        
    def set_page_budget(self, page_budget: int):
        self._page_budget = page_budget
        
    def prepare_metadata(self, seq_len: int):
        """
        Allocates pages in KV cache.
        """
        self.kv_cache.append_seq(seq_len)
        
    def update_metadata(self, layer_idx: int, page_idx: int, k_min: mx.array, k_max: mx.array):
        """
        Updates metadata for a specific page.
        """
        # k_min, k_max: (num_kv_heads, head_dim) - MLX arrays
        k_min_np = np.array(k_min)
        k_max_np = np.array(k_max)
        
        self.metadata_pool[layer_idx, page_idx, 0] = k_min_np
        self.metadata_pool[layer_idx, page_idx, 1] = k_max_np

    def get_metadata_tensor(self, layer_idx: int, page_indices: List[int]) -> mx.array:
        """
        Returns metadata for selected pages as MLX tensor.
        """
        if not page_indices:
            return mx.array([])
        
        data = self.metadata_pool[layer_idx, page_indices]
        # Transpose to (2, NumPages, H_kv, D) to match expected logic
        data = data.transpose(1, 0, 2, 3)
        return mx.array(data)
        
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
