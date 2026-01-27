import time
import mlx.core as mx
import numpy as np
import mlx.nn as nn
from typing import List, Optional, Tuple, Union
from .kv_cache import QuestController

def apply_rope_in_place(
    q: mx.array,
    k: mx.array,
    past_kv_len: int,
    rope_scale: float = 1.0,
    rope_theta: float = 1e4,
):
    """
    Apply RoPE in-place (conceptually).
    This is a simplified implementation placeholder.
    """
    return q, k

def append_kv(
    k: mx.array,
    v: mx.array,
    controller: QuestController,
    layer_idx: int
):
    """
    Appends K/V to the cache and updates metadata.
    k, v: (seq_len, num_kv_heads, head_dim)
    """
    seq_len = k.shape[0]
    
    start_idx = controller.kv_cache.seq_len - seq_len
    
    remaining = seq_len
    current_offset_in_input = 0
    
    while remaining > 0:
        abs_idx = start_idx + current_offset_in_input
        page_idx = abs_idx // controller.page_size
        page_offset = abs_idx % controller.page_size
        
        space_in_page = controller.page_size - page_offset
        tokens_to_write = min(remaining, space_in_page)
        
        # Get data slice
        k_chunk = k[current_offset_in_input : current_offset_in_input + tokens_to_write]
        v_chunk = v[current_offset_in_input : current_offset_in_input + tokens_to_write]
        
        k_chunk_f32 = k_chunk.astype(mx.float32)
        v_chunk_f32 = v_chunk.astype(mx.float32)
        
        is_last_page = (page_idx == len(controller.kv_cache.active_indices) - 1)
        
        if is_last_page:
            buffer = controller.kv_cache.get_active_buffer(layer_idx)
            buffer_mx = controller.kv_cache.get_active_buffer_mx(layer_idx)
            # buffer shape: (2, page_size, H_kv, D)
            k_np = np.array(k_chunk_f32)
            v_np = np.array(v_chunk_f32)
            k_np_disk = k_np.astype(buffer.dtype, copy=False)
            v_np_disk = v_np.astype(buffer.dtype, copy=False)
            buffer[0, page_offset : page_offset + tokens_to_write] = k_np_disk
            buffer[1, page_offset : page_offset + tokens_to_write] = v_np_disk
            if buffer_mx is not None:
                buffer_mx[0, page_offset : page_offset + tokens_to_write] = k_chunk.astype(buffer_mx.dtype)
                buffer_mx[1, page_offset : page_offset + tokens_to_write] = v_chunk.astype(buffer_mx.dtype)
            
            # Update Metadata incrementally to avoid rescanning the active page.
            chunk_k_min = np.min(k_np, axis=0)
            chunk_k_max = np.max(k_np, axis=0)
            
            # Use physical index for metadata
            phys_page_idx = controller.kv_cache.active_indices[page_idx]
            if page_offset == 0:
                controller.update_metadata(layer_idx, phys_page_idx, chunk_k_min, chunk_k_max)
            else:
                prev_min = controller.metadata_pool[layer_idx, phys_page_idx, 0]
                prev_max = controller.metadata_pool[layer_idx, phys_page_idx, 1]
                new_min = np.minimum(prev_min, chunk_k_min)
                new_max = np.maximum(prev_max, chunk_k_max)
                controller.update_metadata(layer_idx, phys_page_idx, new_min, new_max)
            
        else:
            physical_block_idx = controller.kv_cache.active_indices[page_idx]
            
            # Write K
            k_np = np.array(k_chunk_f32)
            k_np_disk = k_np.astype(controller.kv_cache.disk_pool.dtype, copy=False)
            controller.kv_cache.disk_pool[layer_idx, physical_block_idx, 0, page_offset : page_offset + tokens_to_write] = k_np_disk
            # Write V
            v_np = np.array(v_chunk_f32)
            v_np_disk = v_np.astype(controller.kv_cache.disk_pool.dtype, copy=False)
            controller.kv_cache.disk_pool[layer_idx, physical_block_idx, 1, page_offset : page_offset + tokens_to_write] = v_np_disk
            
            # Metadata update for disk page
            # Update metadata even for partial writes to non-last pages
            # (e.g. filling the tail of a page that just became full)
            
            chunk_k_min = np.min(k_np, axis=0)
            chunk_k_max = np.max(k_np, axis=0)
            
            if page_offset == 0:
                # First write to this page (or full overwrite), set metadata directly
                controller.update_metadata(layer_idx, physical_block_idx, chunk_k_min, chunk_k_max)
            else:
                # Partial update: Merge with existing metadata
                # metadata_pool shape: (layers, capacity, 2, H, D)
                
                prev_min = controller.metadata_pool[layer_idx][physical_block_idx, 0]
                prev_max = controller.metadata_pool[layer_idx][physical_block_idx, 1]
                
                new_min = np.minimum(prev_min, chunk_k_min)
                new_max = np.maximum(prev_max, chunk_k_max)
                
                controller.update_metadata(layer_idx, physical_block_idx, new_min, new_max)
        
        current_offset_in_input += tokens_to_write
        remaining -= tokens_to_write


def decode_estimate(
    q: mx.array,
    controller: QuestController,
    layer_idx: int
) -> mx.array:
    """
    Estimate attention scores using in-memory metadata.
    q: (1, H_q, D)
    """
    # 1. Get metadata for candidate pages
    pages_indices = controller.kv_indices_without_last
    if not pages_indices:
        return mx.zeros((q.shape[1], 0), dtype=q.dtype)
        
    # Get metadata tensor: (2, NumPages, H_kv, D)
    metadata = controller.get_metadata_tensor(layer_idx, pages_indices)
    
    K_min = metadata[0] # (NumPages, H_kv, D) -> need (H_kv, NumPages, D)
    K_max = metadata[1]
    
    K_min = K_min.transpose(1, 0, 2)
    K_max = K_max.transpose(1, 0, 2)
    
    # Check for GQA mismatch
    H_q = q.shape[1]
    H_kv = K_min.shape[0]
    
    if H_q != H_kv:
        n_rep = H_q // H_kv
        # Expand KV metadata to match Query heads
        # (H_kv, NumPages, D) -> (H_q, NumPages, D)
        K_min = mx.repeat(K_min, n_rep, axis=0)
        K_max = mx.repeat(K_max, n_rep, axis=0)
    
    # Q: (1, H_q, D) -> (H_q, 1, D)
    Q = q.transpose(1, 0, 2)
    
    Q_pos = mx.maximum(Q, 0.0)
    Q_neg = mx.minimum(Q, 0.0)
    
    term1 = Q_pos * K_max 
    term2 = Q_neg * K_min 
    
    score = mx.sum(term1 + term2, axis=-1) # (H_q, NumPages)
    
    return score


def decode_topk(
    estimated_scores: mx.array,
    page_budget: int
) -> mx.array:
    """
    Select top-k pages.
    """
    if estimated_scores.shape[1] <= page_budget:
        base_indices = mx.arange(estimated_scores.shape[1])[None, :]
        return mx.repeat(base_indices, estimated_scores.shape[0], axis=0)
        
    k = page_budget
    indices = mx.argpartition(estimated_scores, -k, axis=-1)
    top_indices = indices[:, -k:]
    
    # Sort indices to maintain chronological order of KV pages
    top_indices = mx.sort(top_indices, axis=-1)
    
    return top_indices


def decode_sparse_attn(
    q: mx.array,
    topk_indices: mx.array,
    controller: QuestController,
    layer_idx: int,
    rope_scale: float = 1.0,
    rope_theta: float = 1e4
) -> mx.array:
    """
    Compute attention using selected pages (from Disk) + last page (active buffer).
    q: (1, H_q, 1, D)
    topk_indices: (H_q, k)
    """
    timing_hook = getattr(controller, "_timing_hook", None)
    
    # Gather Logic for GQA
    # Gather KV pages based on topk_indices.
    
    # Force candidate_indices to be int64 to ensure physical_indices inherits int type
    if timing_hook:
        t_indices = time.perf_counter()
    candidate_indices = np.array(controller.kv_indices_without_last, dtype=np.int64)
    topk_indices_np = np.array(topk_indices).astype(np.int64) # (H_q, k)
    logical_indices = candidate_indices[topk_indices_np]
    active_indices = np.array(controller.kv_cache.active_indices, dtype=np.int64)
    physical_indices = active_indices[logical_indices].astype(np.int64)
    if timing_hook:
        timing_hook("decode_indices", t_indices)
    
    # Pre-load last page (from active buffer)
    if timing_hook:
        t_last = time.perf_counter()
    active_buffer = controller.kv_cache.get_active_buffer(layer_idx)
    active_buffer_mx = controller.kv_cache.get_active_buffer_mx(layer_idx)
    valid_len = controller.kv_cache.last_page_len
    
    # Active buffer is (2, page_size, H_kv, D)
    if active_buffer_mx is not None:
        last_k_mx = active_buffer_mx[0, :valid_len].astype(q.dtype)
        last_v_mx = active_buffer_mx[1, :valid_len].astype(q.dtype)
    else:
        last_k_mx = mx.array(active_buffer[0, :valid_len]).astype(q.dtype) # (Len, H_kv, D)
        last_v_mx = mx.array(active_buffer[1, :valid_len]).astype(q.dtype)
    
    num_heads = q.shape[1] # H_q
    num_kv_heads = controller.kv_cache.num_heads # H_kv
    group_size = num_heads // num_kv_heads
    
    # 1. Vectorized Disk Load (LRU cached)
    # physical_indices: (H_q, k)
    # kv_head_indices: (H_q, 1)
    if timing_hook:
        t_disk = time.perf_counter()
    kv_head_indices = (np.arange(num_heads, dtype=np.int64) // group_size)[:, None]
    pages_np = controller.kv_cache.load_kv_slices(
        layer_idx,
        physical_indices,
        kv_head_indices,
        timing_hook=timing_hook
    )
    if timing_hook:
        timing_hook("decode_disk_read", t_disk)
    
    # Separate K/V and Flatten
    # pages_np: (H_q, k, 2, PageSize, D)
    k_pages = pages_np[:, :, 0, :, :] # (H_q, k, PageSize, D)
    v_pages = pages_np[:, :, 1, :, :]
    
    # Reshape to (H_q, TotalDiskLen, D)
    k_disk = k_pages.reshape(num_heads, -1, controller.head_dim)
    v_disk = v_pages.reshape(num_heads, -1, controller.head_dim)
    
    # Convert to MLX
    if timing_hook:
        t_disk_mx = time.perf_counter()
    k_disk_mx = mx.array(k_disk).astype(q.dtype)
    v_disk_mx = mx.array(v_disk).astype(q.dtype)
    if timing_hook:
        timing_hook("decode_disk_to_mx", t_disk_mx, k_disk_mx, v_disk_mx)
    
    # 2. Process Last Page (Active Buffer)
    # last_k_mx: (Len, H_kv, D) -> (H_kv, Len, D)
    last_k_mx = last_k_mx.transpose(1, 0, 2)
    last_v_mx = last_v_mx.transpose(1, 0, 2)
    
    # Expand to H_q
    if group_size > 1:
        last_k_mx = mx.repeat(last_k_mx, group_size, axis=0) # (H_q, Len, D)
        last_v_mx = mx.repeat(last_v_mx, group_size, axis=0)
    if timing_hook:
        timing_hook("decode_last_page", t_last, last_k_mx, last_v_mx)
    
    # 3. Concatenate
    # k_disk_mx: (H_q, DiskLen, D)
    # last_k_mx: (H_q, ActiveLen, D)
    
    if timing_hook:
        t_concat = time.perf_counter()
    K = mx.concatenate([k_disk_mx, last_k_mx], axis=1) # (H_q, TotalLen, D)
    V = mx.concatenate([v_disk_mx, last_v_mx], axis=1)
    
    # Add Batch Dim -> (1, H_q, TotalLen, D)
    K = mx.expand_dims(K, axis=0)
    V = mx.expand_dims(V, axis=0)
    if timing_hook:
        timing_hook("decode_concat", t_concat, K, V)
    
    # Attention
    if timing_hook:
        t_sdpa = time.perf_counter()
    out = mx.fast.scaled_dot_product_attention(q, K, V, scale=1.0/mx.sqrt(q.shape[-1]))
    if timing_hook:
        timing_hook("decode_sdpa", t_sdpa, out)
    
    return out
