import mlx.core as mx
import numpy as np
import os
import shutil
import queue
import threading
from typing import List, Optional, Set, Tuple, Dict

class DiskOffloadKvCache:
    """
    Disk-offloaded KV Cache using numpy.memmap.
    Stores pages in a monolithic file on disk, mapping them into memory as needed.
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
        async_disk_write: bool = False
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
        
        # Initialize memmap
        # mode='w+' creates or overwrites the file
        self.disk_pool = np.memmap(
            self.backing_file,
            dtype=self.dtype,
            mode='w+',
            shape=self.shape
        )

        self._disk_write_queue: Optional[queue.Queue] = None
        self._disk_write_thread: Optional[threading.Thread] = None
        if self.async_disk_write:
            self._disk_write_queue = queue.Queue()
            self._disk_write_thread = threading.Thread(
                target=self._disk_write_worker,
                daemon=True
            )
            self._disk_write_thread.start()
        
        # Track free blocks (simple stack allocator)
        self.free_slots = set(range(self.capacity))
        
        # Active page indices (ordered list of pages used by the sequence)
        self.active_indices: List[int] = []
        
        # Active buffers in RAM: layer_idx -> np.array
        # These hold the current page being filled to avoid frequent small writes to disk
        self.active_buffers: Dict[int, np.ndarray] = {}
        # Active buffers in MX: layer_idx -> mx.array (mirrors active_buffers)
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
                
                # Initialize active buffers for all layers
                for l in range(self.num_layers):
                    # Shape: (2, page_size, num_heads, head_dim)
                    self.active_buffers[l] = np.zeros(
                        (2, self.page_size, self.num_heads, self.head_dim),
                        dtype=self.dtype
                    )
                    self.active_buffers_mx[l] = mx.zeros(
                        (2, self.page_size, self.num_heads, self.head_dim),
                        dtype=self.mx_dtype
                    )
            self.seq_len += 1
        return appended_page_count
    
    def flush_active_buffers(self, page_idx: int):
        """Writes active buffers to disk pool."""
        for l, buffer in self.active_buffers.items():
            self._write_disk_pool(l, page_idx, buffer)
            
    def get_active_buffer(self, layer_idx: int) -> np.ndarray:
        return self.active_buffers[layer_idx]

    def get_active_buffer_mx(self, layer_idx: int) -> Optional[mx.array]:
        return self.active_buffers_mx.get(layer_idx)

    def load_pages(self, layer_idx: int, page_indices: List[int]) -> mx.array:
        """
        Load multiple pages from disk pool.
        Returns mx.array of shape (num_pages, 2, page_size, num_heads, head_dim)
        """
        if not page_indices:
            return mx.array([])
            
        # Convert indices to list for slicing if not already
        # numpy fancy indexing
        pages_np = self.disk_pool[layer_idx, page_indices]
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
        return self.disk_pool[layer_idx, physical_indices, :, :, kv_head_indices, :]

    def release(self):
        if self._disk_write_queue is not None:
            self._disk_write_queue.join()
        self.seq_len = 0
        # Reset free slots
        self.free_slots = set(range(self.capacity))
        self.active_indices.clear()
        self.active_buffers.clear()
        self.active_buffers_mx.clear()

    def _write_disk_pool(self, layer_idx: int, page_idx: int, buffer_np: np.ndarray):
        if self._disk_write_queue is None:
            self.disk_pool[layer_idx, page_idx] = buffer_np
            return
        self._disk_write_queue.put((layer_idx, page_idx, buffer_np))

    def _disk_write_worker(self):
        if self._disk_write_queue is None:
            return
        while True:
            item = self._disk_write_queue.get()
            if item is None:
                self._disk_write_queue.task_done()
                break
            layer_idx, page_idx, buffer_np = item
            self.disk_pool[layer_idx, page_idx] = buffer_np
            self._disk_write_queue.task_done()


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
        async_disk_write: bool = False
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self._page_budget = page_budget
        self.dtype = dtype
        
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
            async_disk_write=async_disk_write
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
