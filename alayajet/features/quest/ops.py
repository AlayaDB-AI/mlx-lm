import os
import time
import mlx.core as mx
import numpy as np
from .kv_cache import QuestController, safe_to_numpy
from .metal_sparse_attention import fused_sparse_decode_attention

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
        
        if tokens_to_write == 1:
            # Min/max over a single token is the token itself; avoid reductions.
            k_chunk_f32 = k_chunk.astype(mx.float32)
            chunk_k = k_chunk_f32[0]
            chunk_k_min = chunk_k
            chunk_k_max = chunk_k
        else:
            k_chunk_f32 = k_chunk.astype(mx.float32)
            chunk_k_min = mx.min(k_chunk_f32, axis=0)
            chunk_k_max = mx.max(k_chunk_f32, axis=0)
        
        is_last_page = (page_idx == len(controller.kv_cache.active_indices) - 1)
        
        if is_last_page:
            buffer_mx = controller.kv_cache.get_active_buffer_mx(layer_idx)
            if buffer_mx is None:
                raise RuntimeError("Active MX buffer missing for last page.")
            buffer_mx[0, page_offset : page_offset + tokens_to_write] = k_chunk.astype(buffer_mx.dtype)
            buffer_mx[1, page_offset : page_offset + tokens_to_write] = v_chunk.astype(buffer_mx.dtype)
            
            # Update Metadata incrementally to avoid rescanning the active page.
            # Use physical index for metadata
            phys_page_idx = controller.kv_cache.active_indices[page_idx]
            if page_offset == 0:
                controller.update_metadata(layer_idx, phys_page_idx, chunk_k_min, chunk_k_max)
            else:
                prev_min = controller.metadata_pool[layer_idx, phys_page_idx, 0]
                prev_max = controller.metadata_pool[layer_idx, phys_page_idx, 1]
                chunk_k_min_np = safe_to_numpy(chunk_k_min, dtype=np.float32)
                chunk_k_max_np = safe_to_numpy(chunk_k_max, dtype=np.float32)
                new_min = np.minimum(prev_min, chunk_k_min_np)
                new_max = np.maximum(prev_max, chunk_k_max_np)
                controller.update_metadata(layer_idx, phys_page_idx, new_min, new_max)
            
        else:
            physical_block_idx = controller.kv_cache.active_indices[page_idx]
            
            k_np = safe_to_numpy(k_chunk, dtype=controller.kv_cache.dtype)
            v_np = safe_to_numpy(v_chunk, dtype=controller.kv_cache.dtype)
            controller.kv_cache.write_kv_slice(
                layer_idx=layer_idx,
                page_idx=physical_block_idx,
                page_offset=page_offset,
                k_np=k_np,
                v_np=v_np,
                assume_zero=(page_offset == 0),
                write_through=controller.prefill_write_through,
            )
            
            # Metadata update for disk page
            # Update metadata even for partial writes to non-last pages
            # (e.g. filling the tail of a page that just became full)
            
            if page_offset == 0:
                # First write to this page (or full overwrite), set metadata directly
                controller.update_metadata(layer_idx, physical_block_idx, chunk_k_min, chunk_k_max)
            else:
                # Partial update: Merge with existing metadata
                # metadata_pool shape: (layers, capacity, 2, H, D)
                
                prev_min = controller.metadata_pool[layer_idx][physical_block_idx, 0]
                prev_max = controller.metadata_pool[layer_idx][physical_block_idx, 1]
                
                chunk_k_min_np = safe_to_numpy(chunk_k_min, dtype=np.float32)
                chunk_k_max_np = safe_to_numpy(chunk_k_max, dtype=np.float32)
                new_min = np.minimum(prev_min, chunk_k_min_np)
                new_max = np.maximum(prev_max, chunk_k_max_np)
                
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
        
    H_q = q.shape[1]
    # Get metadata shaped for estimate: (H_kv, NumPages, D)
    K_min, K_max = controller.get_metadata_kminmax(layer_idx)
    
    H_kv = K_min.shape[0]
    # Q: (1, H_q, D) -> (H_q, 1, D)
    Q = q.transpose(1, 0, 2)

    if H_q == H_kv:
        Q_pos = mx.maximum(Q, 0.0)
        Q_neg = mx.minimum(Q, 0.0)
        term1 = Q_pos * K_max
        term2 = Q_neg * K_min
        score = mx.sum(term1 + term2, axis=-1)  # (H_q, NumPages)
        return score

    group_size = H_q // H_kv
    Q = Q.reshape(H_kv, group_size, 1, Q.shape[-1])
    Q_pos = mx.maximum(Q, 0.0)
    Q_neg = mx.minimum(Q, 0.0)
    K_min = mx.expand_dims(K_min, axis=1)
    K_max = mx.expand_dims(K_max, axis=1)
    score = mx.sum(Q_pos * K_max + Q_neg * K_min, axis=-1)  # (H_kv, group_size, NumPages)
    score = score.reshape(H_q, -1)
    
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


def decode_topk_np(
    estimated_scores: np.ndarray,
    page_budget: int,
) -> np.ndarray:
    """
    Select top-k pages (numpy path).
    """
    if estimated_scores.shape[1] <= page_budget:
        base_indices = np.arange(estimated_scores.shape[1], dtype=np.int64)[None, :]
        return np.repeat(base_indices, estimated_scores.shape[0], axis=0)

    k = page_budget
    indices = np.argpartition(estimated_scores, -k, axis=-1)
    top_indices = indices[:, -k:]
    top_indices = np.sort(top_indices, axis=-1)
    return top_indices.astype(np.int64, copy=False)


def decode_sparse_attn(
    q: mx.array,
    topk_indices: mx.array,
    controller: QuestController,
    layer_idx: int,
    rope_scale: float = 1.0,
    rope_theta: float = 1e4,
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
    topk_indices_np = np.asarray(topk_indices, dtype=np.int64) # (H_q, k)
    logical_indices = candidate_indices[topk_indices_np]
    active_indices = np.array(controller.kv_cache.active_indices, dtype=np.int64)
    physical_indices = active_indices[logical_indices].astype(np.int64)
    if timing_hook:
        timing_hook("decode_indices", t_indices)
    
    num_heads = q.shape[1] # H_q
    num_kv_heads = controller.kv_cache.num_heads # H_kv
    group_size = num_heads // num_kv_heads

    if (
        os.environ.get("ALAYAJET_QUEST_METAL_SPARSE") == "1"
        and controller.need_estimate()
        and len(controller.kv_indices_without_last) > 0
    ):
        active_indices = np.asarray(controller.kv_cache.active_indices, dtype=np.int64)
        candidate_logical = np.asarray(controller.kv_indices_without_last, dtype=np.int64)
        candidate_physical = active_indices[candidate_logical]
        cache_key = (layer_idx, tuple(int(x) for x in candidate_physical.tolist()))
        cached = controller._metal_sparse_cache.get(cache_key)
        if cached is None:
            for stale_key in list(controller._metal_sparse_cache):
                if stale_key[0] == layer_idx:
                    del controller._metal_sparse_cache[stale_key]
            metadata = controller.metadata_pool[layer_idx, candidate_physical]
            k_min = mx.array(metadata[:, 0].transpose(1, 0, 2)).astype(mx.float32)
            k_max = mx.array(metadata[:, 1].transpose(1, 0, 2)).astype(mx.float32)
            kv_pages = controller.kv_cache.load_pages(
                layer_idx,
                candidate_physical.tolist(),
            ).astype(q.dtype)
            controller._metal_sparse_cache[cache_key] = (k_min, k_max, kv_pages)
        else:
            k_min, k_max, kv_pages = cached

        active_buffer_mx = controller.kv_cache.get_active_buffer_mx(layer_idx)
        valid_len = controller.kv_cache.last_page_len
        if active_buffer_mx is None:
            raise RuntimeError("Active MX buffer missing during fused Metal sparse decode.")
        last_k_mx = active_buffer_mx[0, :valid_len].astype(q.dtype)
        last_v_mx = active_buffer_mx[1, :valid_len].astype(q.dtype)
        fused_page_budget = controller.inference_page_budget
        fused_budget_override = os.environ.get("ALAYAJET_QUEST_METAL_PAGE_BUDGET")
        if fused_budget_override:
            fused_page_budget = min(fused_page_budget, int(fused_budget_override))
        out = fused_sparse_decode_attention(
            q,
            k_min,
            k_max,
            kv_pages,
            last_k_mx,
            last_v_mx,
            page_budget=fused_page_budget,
            scale=float(1.0 / np.sqrt(q.shape[-1])),
        )
        return out
    
    kv_head_indices = (np.arange(num_heads, dtype=np.int64) // group_size)[:, None]
    
    # 2. Process Last Page (Active Buffer)
    if timing_hook:
        t_last = time.perf_counter()
    active_buffer_mx = controller.kv_cache.get_active_buffer_mx(layer_idx)
    valid_len = controller.kv_cache.last_page_len
    
    # Active buffer is (2, page_size, H_kv, D)
    if active_buffer_mx is None:
        raise RuntimeError("Active MX buffer missing during decode.")
    last_k_mx = active_buffer_mx[0, :valid_len].astype(q.dtype)
    last_v_mx = active_buffer_mx[1, :valid_len].astype(q.dtype)

    # last_k_mx: (Len, H_kv, D) -> (H_kv, Len, D)
    last_k_mx = last_k_mx.transpose(1, 0, 2)
    last_v_mx = last_v_mx.transpose(1, 0, 2)
    
    # Expand to H_q
    if group_size > 1:
        last_k_mx = mx.repeat(last_k_mx, group_size, axis=0) # (H_q, Len, D)
        last_v_mx = mx.repeat(last_v_mx, group_size, axis=0)
    if timing_hook:
        timing_hook("decode_last_page", t_last, last_k_mx, last_v_mx)
    
    # 3. One-shot SDPA (streaming path removed).
    if timing_hook:
        t_stream = time.perf_counter()
    scale = 1.0 / mx.sqrt(q.shape[-1])
    if timing_hook:
        t_disk = time.perf_counter()
    disk_len = int(topk_indices.shape[1]) * controller.page_size
    k_disk_mx, v_disk_mx = controller.kv_cache.load_kv_slices_mx(
        layer_idx,
        physical_indices,
        kv_head_indices,
        dtype=q.dtype,
    )
    if timing_hook:
        timing_hook("decode_disk_read", t_disk)
    if disk_len == 0 and valid_len == 0:
        out = mx.zeros_like(q)
    else:
        if disk_len == 0:
            k_full = last_k_mx
            v_full = last_v_mx
        elif valid_len == 0:
            k_full = k_disk_mx
            v_full = v_disk_mx
        else:
            k_full = mx.concatenate([k_disk_mx, last_k_mx], axis=1)
            v_full = mx.concatenate([v_disk_mx, last_v_mx], axis=1)
        k_full = mx.expand_dims(k_full, axis=0)
        v_full = mx.expand_dims(v_full, axis=0)
        out = mx.fast.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            scale=scale,
        )
    if timing_hook:
        timing_hook("decode_stream_attn", t_stream, out)
    return out
