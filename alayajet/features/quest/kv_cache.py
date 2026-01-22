import mlx.core as mx
import numpy as np
import os
import shutil
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
        cache_dir: str = "./kv_cache_tmp",
        name: str = "kv_cache"
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads # Stores num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = dtype
        self.name = name
        self.cache_dir = cache_dir

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
        
        # Track free blocks (simple stack allocator)
        self.free_slots = set(range(self.capacity))
        
        # Active page indices (ordered list of pages used by the sequence)
        self.active_indices: List[int] = []
        
        # Active buffers in RAM: layer_idx -> np.array or mx.array
        # These hold the current page being filled to avoid frequent small writes to disk
        self.active_buffers: Dict[int, np.ndarray] = {}
        
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
            self.seq_len += 1
        return appended_page_count
    
    def flush_active_buffers(self, page_idx: int):
        """Writes active buffers to disk pool."""
        for l, buffer in self.active_buffers.items():
            self.disk_pool[l, page_idx] = buffer
            
    def get_active_buffer(self, layer_idx: int) -> np.ndarray:
        return self.active_buffers[layer_idx]

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

    def release(self):
        self.seq_len = 0
        # Reset free slots
        self.free_slots = set(range(self.capacity))
        self.active_indices.clear()
        self.active_buffers.clear()


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
        cache_dir: str = "./kv_cache_tmp"
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self._page_budget = page_budget
        self.dtype = dtype
        
        # Main KV Cache (Disk Backed)
        # Note: Use float32 for disk cache to safely store float16 and bfloat16
        np_dtype = np.float32
        
        self.kv_cache = DiskOffloadKvCache(
            num_layers=num_layers,
            num_heads=self.num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            page_size=page_size,
            dtype=np_dtype,
            cache_dir=cache_dir,
            name="kv_cache"
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
        # Convert to numpy
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
        
        # Fancy indexing with numpy
        data = self.metadata_pool[layer_idx, page_indices] # (NumPages, 2, H_kv, D)
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
