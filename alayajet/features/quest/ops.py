import os
import time
import mlx.core as mx
import numpy as np
from .kv_cache import QuestController, safe_to_numpy
from .metal_sparse_attention import (
    fused_sparse_decode_attention,
    pack_kv_pages_head_major,
    pack_last_page_head_major,
    selected_direct_frame_decode_attention,
    selected_frame_decode_attention,
    selected_frame_decode_attention_with_last_frame,
    selected_frame_decode_attention_with_stable_arena,
    selected_indexed_page_decode_attention,
)

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
    resident_write_mode = os.environ.get("ALAYAJET_QUEST_RESIDENT_FRAME_WRITE", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    resident_decode_write = resident_write_mode and seq_len == 1
    
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
            phys_page_idx = controller.kv_cache.active_indices[page_idx]
            if resident_decode_write:
                buffer_mx = controller.kv_cache.get_active_buffer_mx(layer_idx)
                if buffer_mx is None:
                    raise RuntimeError("Active MX buffer missing for last page.")
                buffer_mx[0, page_offset : page_offset + tokens_to_write] = k_chunk.astype(buffer_mx.dtype)
                buffer_mx[1, page_offset : page_offset + tokens_to_write] = v_chunk.astype(buffer_mx.dtype)
                controller.kv_cache.remember_decode_rolling_buffer(
                    layer_idx,
                    phys_page_idx,
                    buffer_mx,
                    dirty=True,
                )
                if controller.resident_frame_pool.has_decode_page(layer_idx, phys_page_idx):
                    controller.resident_frame_pool.write_decode_kv_slices(
                        layer_idx=layer_idx,
                        block_idx=phys_page_idx,
                        page_offset=page_offset,
                        k_mx=k_chunk,
                        v_mx=v_chunk,
                    )
                else:
                    written_frame_ids = controller.resident_frame_pool.write_decode_page(
                        layer_idx=layer_idx,
                        block_idx=phys_page_idx,
                        page_mx=buffer_mx,
                    )
                    end = page_offset + tokens_to_write
                    for frame_id in written_frame_ids:
                        controller.resident_frame_pool.meta[int(frame_id)].valid_length = end
                    controller.resident_frame_pool.decode_page_valid_lengths[layer_idx][
                        phys_page_idx
                    ] = end
                    controller.resident_frame_pool.last_page_valid_length[layer_idx] = end
                if page_offset + tokens_to_write == controller.page_size:
                    controller.kv_cache.flush_decode_rolling_full_pages()
            else:
                buffer_mx = controller.kv_cache.get_active_buffer_mx(layer_idx)
                if buffer_mx is None:
                    raise RuntimeError("Active MX buffer missing for last page.")
                buffer_mx[0, page_offset : page_offset + tokens_to_write] = k_chunk.astype(buffer_mx.dtype)
                buffer_mx[1, page_offset : page_offset + tokens_to_write] = v_chunk.astype(buffer_mx.dtype)
            
            # Update Metadata incrementally to avoid rescanning the active page.
            # Use physical index for metadata
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


def decode_topk_frame_ids_np(
    estimated_scores: np.ndarray,
    page_budget: int,
    controller: QuestController,
    layer_idx: int,
) -> np.ndarray:
    """Select top-k resident KV frames for the latest frame-native path."""
    topk = decode_topk_np(estimated_scores, page_budget)
    return decode_topk_frame_ids_from_indices_np(topk, controller, layer_idx)


def decode_topk_frame_ids_from_indices_np(
    topk_indices: np.ndarray,
    controller: QuestController,
    layer_idx: int,
) -> np.ndarray:
    if topk_indices.size == 0:
        return np.asarray(topk_indices, dtype=np.int64)
    candidate_logical = np.asarray(controller.kv_indices_without_last, dtype=np.int64)
    active_indices = np.asarray(controller.kv_cache.active_indices, dtype=np.int64)
    physical_blocks = active_indices[candidate_logical[np.asarray(topk_indices, dtype=np.int64)]]
    num_heads = physical_blocks.shape[0]
    num_kv_heads = controller.kv_cache.num_heads
    group_size = max(1, num_heads // num_kv_heads)
    frame_ids = np.empty_like(physical_blocks, dtype=np.int64)
    for h in range(num_heads):
        kv_head = h // group_size
        for j, block_idx in enumerate(physical_blocks[h]):
            frame_ids[h, j] = controller.resident_frame_pool.frame_id(
                layer_idx,
                int(block_idx),
                int(kv_head),
            )
    return frame_ids


def _select_cpu_union_candidates(
    q: mx.array,
    metadata: np.ndarray,
    page_budget: int,
    group_size: int,
) -> np.ndarray:
    """Return sorted candidate offsets containing every head's exact top-k pages.

    ``metadata`` is shaped as (num_pages, 2, num_kv_heads, head_dim).  The
    returned offsets index into that metadata/page list, not physical page ids.
    """
    num_pages = int(metadata.shape[0])
    if num_pages == 0:
        return np.empty((0,), dtype=np.int64)
    selected_count = min(int(page_budget), num_pages)
    if selected_count >= num_pages:
        return np.arange(num_pages, dtype=np.int64)

    if q.ndim == 4:
        q_np = safe_to_numpy(q[0, :, 0, :], dtype=np.float32)
    elif q.ndim == 2:
        q_np = safe_to_numpy(q, dtype=np.float32)
    else:
        raise ValueError(f"Expected q rank 2 or 4, got shape {q.shape}")

    k_min = metadata[:, 0].astype(np.float32, copy=False)
    k_max = metadata[:, 1].astype(np.float32, copy=False)
    selected = []
    for head_idx, q_head in enumerate(q_np):
        kv_head = head_idx // group_size
        q_pos = np.maximum(q_head, 0.0)
        q_neg = np.minimum(q_head, 0.0)
        scores = (q_pos * k_max[:, kv_head, :] + q_neg * k_min[:, kv_head, :]).sum(axis=-1)
        top = np.argpartition(scores, -selected_count)[-selected_count:]
        selected.append(top)
    union = np.unique(np.concatenate(selected).astype(np.int64, copy=False))
    union.sort()
    return union


def _use_cpu_union_page_selection(
    mode: str,
    num_pages: int,
    num_heads: int,
    page_budget: int,
) -> bool:
    mode = mode.strip().lower()
    if mode in ("cpu", "cpu_union", "cooperative"):
        return True
    if mode in ("0", "off", "false", "metal", "gpu"):
        return False
    if mode != "auto":
        raise ValueError(
            "ALAYAJET_QUEST_PAGE_SELECT must be one of: metal, cpu_union, auto"
        )

    # A real 32k run on M-series showed that GPU->CPU q synchronization plus
    # NumPy scoring is much more expensive than the fused kernel's metadata scan
    # at a few hundred pages. Keep auto conservative and leave smaller sweeps to
    # explicit ``cpu_union`` experiments.
    threshold = int(os.environ.get("ALAYAJET_QUEST_CPU_SELECT_MIN_PAGES", "4096"))
    min_reduction = float(os.environ.get("ALAYAJET_QUEST_CPU_SELECT_MIN_REDUCTION", "4.0"))
    max_union = max(1, int(num_heads) * int(page_budget))
    return int(num_pages) > max(threshold, int(max_union * min_reduction))


def _use_selected_page_metal_attention() -> bool:
    return os.environ.get("ALAYAJET_QUEST_METAL_SELECTED_PAGE", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _selected_frame_metal_mode() -> str:
    mode = os.environ.get("ALAYAJET_QUEST_METAL_SELECTED_FRAME", "0").lower()
    if mode in ("1", "true", "yes", "on", "compact"):
        return "compact"
    if mode in ("hot", "hot_arena", "selected_hot"):
        return "hot"
    if mode in ("resident", "resident_arena", "global"):
        return "resident"
    if mode in ("layer", "layer_arena"):
        return "layer"
    if mode in ("direct", "direct_frame", "direct_frames"):
        return "direct"
    if mode in ("page_arena", "pagearena", "resident_page", "resident_pages"):
        return "page_arena"
    return "off"


def _use_selected_frame_metal_attention() -> bool:
    return _selected_frame_metal_mode() != "off"


def _topk_is_resident_frame_ids() -> bool:
    return False


def _load_selected_kv_head_pages(
    controller: QuestController,
    layer_idx: int,
    physical_indices: np.ndarray,
    kv_head_indices: np.ndarray,
    dtype,
) -> tuple[mx.array, mx.array]:
    num_heads, selected_count = physical_indices.shape
    num_kv_heads = controller.kv_cache.num_heads
    page_size = controller.page_size
    head_dim = controller.head_dim
    kv_head_indices = np.asarray(kv_head_indices, dtype=np.int64).reshape(num_heads)

    pages_by_kv = []
    offsets = np.zeros((num_heads, selected_count), dtype=np.int32)
    max_pages = 1
    per_kv_pages = []
    for kv_head in range(num_kv_heads):
        query_heads = np.nonzero(kv_head_indices == kv_head)[0]
        pages = np.unique(physical_indices[query_heads]) if query_heads.size else np.empty((0,), dtype=np.int64)
        pages = np.asarray(pages, dtype=np.int64)
        per_kv_pages.append(pages)
        max_pages = max(max_pages, int(pages.size))
        for h in query_heads:
            offsets[h] = np.searchsorted(pages, physical_indices[h]).astype(np.int32)

    for kv_head, pages in enumerate(per_kv_pages):
        head_pages = []
        for page_idx in pages:
            page_id = (layer_idx, int(page_idx), kv_head)
            head_pages.append(controller.kv_cache.buffer_pool.get_page_mx(page_id))
        if head_pages:
            pages_mx = mx.stack(head_pages, axis=0)
            if pages_mx.dtype != dtype:
                pages_mx = pages_mx.astype(dtype)
        else:
            pages_mx = mx.zeros((0, 2, page_size, head_dim), dtype=dtype)
        pad = max_pages - int(pages_mx.shape[0])
        if pad:
            padding = mx.zeros((pad, 2, page_size, head_dim), dtype=dtype)
            pages_mx = mx.concatenate([pages_mx, padding], axis=0)
        pages_by_kv.append(pages_mx)

    kv_pages = mx.stack(pages_by_kv, axis=0)
    return mx.array(offsets), kv_pages


def _load_selected_frame_pages(
    controller: QuestController,
    layer_idx: int,
    physical_indices: np.ndarray,
    kv_head_indices: np.ndarray,
    dtype,
) -> tuple[mx.array, mx.array]:
    physical_indices = np.asarray(physical_indices, dtype=np.int64)
    num_heads, selected_count = physical_indices.shape
    kv_head_indices = np.asarray(kv_head_indices, dtype=np.int64).reshape(num_heads)

    page_ids = []
    for h in range(num_heads):
        kv_head = int(kv_head_indices[h])
        for page_idx in physical_indices[h]:
            page_ids.append((layer_idx, int(page_idx), kv_head))
    unique_page_ids = list(dict.fromkeys(page_ids))

    frame_indices_by_page = {
        page_id: controller.kv_cache.buffer_pool.get_frame_index(page_id)
        for page_id in unique_page_ids
    }
    frame_indices = [frame_indices_by_page[page_id] for page_id in unique_page_ids]
    frame_key = (
        layer_idx,
        tuple(
            (int(frame_idx), int(page_id[1]), int(page_id[2]))
            for frame_idx, page_id in zip(frame_indices, unique_page_ids)
        ),
    )

    frames_mx = controller._metal_selected_frame_cache.get(frame_key)
    if frames_mx is None:
        for stale_key in list(controller._metal_selected_frame_cache):
            if stale_key[0] == layer_idx:
                del controller._metal_selected_frame_cache[stale_key]
        compact_lookup = {
            int(frame_idx): compact_idx
            for compact_idx, frame_idx in enumerate(frame_indices)
        }
        selected_frames = np.empty((num_heads, selected_count), dtype=np.int32)
        for linear_idx, page_id in enumerate(page_ids):
            h = linear_idx // selected_count
            j = linear_idx - h * selected_count
            selected_frames[h, j] = compact_lookup[
                int(frame_indices_by_page[page_id])
            ]
        selected_frames_mx = mx.array(selected_frames)
        frames_mx = controller.kv_cache.buffer_pool.get_frames_mx(frame_indices, dtype=dtype)
        controller._metal_selected_frame_cache[frame_key] = (
            selected_frames_mx,
            frames_mx,
        )
    else:
        selected_frames_mx, frames_mx = frames_mx
    return selected_frames_mx, frames_mx


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
    topk_indices_np = np.asarray(topk_indices, dtype=np.int64) # (H_q, k)
    if _topk_is_resident_frame_ids() and _selected_frame_metal_mode() == "resident":
        num_kv_heads_for_ids = controller.kv_cache.num_heads
        local_frame_ids = topk_indices_np - int(layer_idx) * controller.resident_frame_pool.max_blocks * num_kv_heads_for_ids
        physical_indices = (local_frame_ids // num_kv_heads_for_ids).astype(np.int64)
    else:
        candidate_indices = np.array(controller.kv_indices_without_last, dtype=np.int64)
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
        and not _use_selected_page_metal_attention()
        and controller.need_estimate()
        and len(controller.kv_indices_without_last) > 0
    ):
        active_indices = np.asarray(controller.kv_cache.active_indices, dtype=np.int64)
        candidate_logical = np.asarray(controller.kv_indices_without_last, dtype=np.int64)
        candidate_physical = active_indices[candidate_logical]
        metal_kv_layout = os.environ.get("ALAYAJET_QUEST_METAL_KV_LAYOUT", "page_major")
        fused_page_budget = controller.inference_page_budget
        fused_budget_override = os.environ.get("ALAYAJET_QUEST_METAL_PAGE_BUDGET")
        if fused_budget_override:
            fused_page_budget = min(fused_page_budget, int(fused_budget_override))
        page_select_mode = os.environ.get("ALAYAJET_QUEST_PAGE_SELECT", "metal")
        if _use_cpu_union_page_selection(
            page_select_mode,
            len(candidate_physical),
            num_heads,
            fused_page_budget,
        ):
            if timing_hook:
                t_cpu_select = time.perf_counter()
            full_metadata = controller.metadata_pool[layer_idx, candidate_physical]
            union_offsets = _select_cpu_union_candidates(
                q,
                full_metadata,
                fused_page_budget,
                group_size,
            )
            if 0 < len(union_offsets) < len(candidate_physical):
                candidate_physical = candidate_physical[union_offsets]
            if timing_hook:
                timing_hook("decode_cpu_page_select", t_cpu_select)
        cache_key = (
            layer_idx,
            metal_kv_layout,
            tuple(int(x) for x in candidate_physical.tolist()),
        )
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
            if metal_kv_layout == "head_major":
                kv_pages = pack_kv_pages_head_major(kv_pages)
            elif metal_kv_layout != "page_major":
                raise ValueError(
                    "ALAYAJET_QUEST_METAL_KV_LAYOUT must be 'page_major' or 'head_major'"
                )
            controller._metal_sparse_cache[cache_key] = (k_min, k_max, kv_pages)
        else:
            k_min, k_max, kv_pages = cached

        active_buffer_mx = controller.kv_cache.get_active_buffer_mx(layer_idx)
        valid_len = controller.kv_cache.last_page_len
        if active_buffer_mx is None:
            raise RuntimeError("Active MX buffer missing during fused Metal sparse decode.")
        if metal_kv_layout == "head_major":
            last_k_mx = pack_last_page_head_major(
                active_buffer_mx[0, :valid_len].astype(q.dtype)
            )
            last_v_mx = pack_last_page_head_major(
                active_buffer_mx[1, :valid_len].astype(q.dtype)
            )
            last_layout = "head_major"
        else:
            last_k_mx = active_buffer_mx[0, :valid_len].astype(q.dtype)
            last_v_mx = active_buffer_mx[1, :valid_len].astype(q.dtype)
            last_layout = "token_major"
        out = fused_sparse_decode_attention(
            q,
            k_min,
            k_max,
            kv_pages,
            last_k_mx,
            last_v_mx,
            page_budget=fused_page_budget,
            scale=float(1.0 / np.sqrt(q.shape[-1])),
            kv_layout=metal_kv_layout,
            last_layout=last_layout,
        )
        return out
    
    kv_head_indices = (np.arange(num_heads, dtype=np.int64) // group_size)[:, None]
    
    # 2. Process Last Page (Active Buffer)
    if timing_hook:
        t_last = time.perf_counter()
    valid_len = controller.kv_cache.last_page_len
    disk_len = int(topk_indices.shape[1]) * controller.page_size
    selected_frame_mode = _selected_frame_metal_mode()
    use_selected_metal = _use_selected_page_metal_attention() and (
        disk_len > 0 or (selected_frame_mode == "resident" and valid_len > 0)
    )
    use_resident_frame_pool = use_selected_metal and selected_frame_mode == "resident"
    
    last_k_token_mx = None
    last_v_token_mx = None
    if not use_resident_frame_pool:
        # Active buffer is (2, page_size, H_kv, D)
        active_buffer_mx = controller.kv_cache.get_active_buffer_mx(layer_idx)
        if active_buffer_mx is None:
            raise RuntimeError("Active MX buffer missing during decode.")
        last_k_token_mx = active_buffer_mx[0, :valid_len].astype(q.dtype)
        last_v_token_mx = active_buffer_mx[1, :valid_len].astype(q.dtype)
    # 3. One-shot SDPA (streaming path removed).
    if timing_hook:
        t_stream = time.perf_counter()
    scale = 1.0 / mx.sqrt(q.shape[-1])
    if timing_hook:
        t_disk = time.perf_counter()
    last_k_mx = None
    last_v_mx = None
    if not use_selected_metal:
        if last_k_token_mx is None or last_v_token_mx is None:
            raise RuntimeError("Active MX buffer missing during decode.")
        # last_k_mx: (Len, H_kv, D) -> (H_kv, Len, D)
        last_k_mx = last_k_token_mx.transpose(1, 0, 2)
        last_v_mx = last_v_token_mx.transpose(1, 0, 2)
        # Expand only for the fallback SDPA path. The selected Metal kernels
        # read GQA last-page heads directly, so repeating here is wasted work.
        if group_size > 1:
            last_k_mx = mx.repeat(last_k_mx, group_size, axis=0)
            last_v_mx = mx.repeat(last_v_mx, group_size, axis=0)
        if timing_hook:
            timing_hook("decode_last_page", t_last, last_k_mx, last_v_mx)
    elif timing_hook and not use_resident_frame_pool:
        timing_hook("decode_last_page", t_last, last_k_token_mx, last_v_token_mx)
    elif timing_hook:
        timing_hook("decode_last_page", t_last)

    if disk_len == 0 and valid_len == 0:
        out = mx.zeros_like(q)
    elif use_selected_metal:
        controller.kv_cache.mark_selected_kv_pages(
            layer_idx,
            physical_indices,
            kv_head_indices,
        )
        selected_physical = np.unique(physical_indices)
        metal_kv_layout = os.environ.get("ALAYAJET_QUEST_METAL_KV_LAYOUT", "head_major")
        if selected_frame_mode != "off":
            if selected_frame_mode == "page_arena":
                arena_layout = os.environ.get(
                    "ALAYAJET_QUEST_PAGE_ARENA_LAYOUT",
                    "head_major",
                )
                _, kv_pages, selected_kv_layout = controller.kv_cache.get_layer_page_arena_mx(
                    layer_idx,
                    selected_physical.tolist(),
                    dtype=q.dtype,
                    layout=arena_layout,
                )
                index_key = (
                    layer_idx,
                    tuple(int(x) for x in physical_indices.reshape(-1).tolist()),
                    tuple(int(x) for x in kv_head_indices.reshape(-1).tolist()),
                )
                selected_pages_mx = controller._metal_page_arena_index_cache.get(index_key)
                if selected_pages_mx is None:
                    for stale_key in list(controller._metal_page_arena_index_cache):
                        if stale_key[0] == layer_idx:
                            del controller._metal_page_arena_index_cache[stale_key]
                    selected_pages_mx = mx.array(
                        np.asarray(physical_indices, dtype=np.int32)
                    )
                    controller._metal_page_arena_index_cache[index_key] = selected_pages_mx
                if selected_kv_layout == "head_major":
                    last_k_selected = pack_last_page_head_major(last_k_token_mx)
                    last_v_selected = pack_last_page_head_major(last_v_token_mx)
                    last_layout = "head_major"
                else:
                    last_k_selected = last_k_token_mx
                    last_v_selected = last_v_token_mx
                    last_layout = "token_major"
                if timing_hook:
                    timing_hook("decode_disk_read", t_disk, kv_pages)
                if timing_hook:
                    t_selected_metal = time.perf_counter()
                out = selected_indexed_page_decode_attention(
                    q,
                    selected_pages_mx,
                    kv_pages,
                    last_k_selected,
                    last_v_selected,
                    scale=float(1.0 / np.sqrt(q.shape[-1])),
                    kv_layout=selected_kv_layout,
                    last_layout=last_layout,
                )
                if timing_hook:
                    timing_hook("decode_selected_page_metal_attn", t_selected_metal, out)
                return out
            if selected_frame_mode == "layer":
                selected_frames_mx, frames_mx = (
                    controller.kv_cache.load_selected_layer_frame_indices_mx(
                        layer_idx,
                        physical_indices,
                        kv_head_indices,
                        q.dtype,
                    )
                )
            elif selected_frame_mode == "hot":
                selected_frames_mx, frames_mx = (
                    controller.kv_cache.load_selected_hot_frame_indices_mx(
                        layer_idx,
                        physical_indices,
                        kv_head_indices,
                        q.dtype,
                    )
                )
            elif selected_frame_mode == "resident":
                kv_head_indices_1d = np.asarray(kv_head_indices, dtype=np.int64).reshape(-1)
                for block_idx in np.unique(physical_indices):
                    block_idx = int(block_idx)
                    if controller.resident_frame_pool.has_decode_page(layer_idx, block_idx):
                        continue
                    if not controller.resident_frame_pool.has_page(layer_idx, block_idx):
                        page_mx = controller.kv_cache.read_page_mx_from_disk(
                            layer_idx,
                            block_idx,
                            dtype=q.dtype,
                        )
                        controller.resident_frame_pool.write_page(
                            layer_idx,
                            block_idx,
                            page_mx,
                        )
                selected_frame_ids = np.empty_like(physical_indices, dtype=np.int64)
                for h in range(physical_indices.shape[0]):
                    kv_head = int(kv_head_indices_1d[h])
                    for j, block_idx in enumerate(physical_indices[h]):
                        frame_id = controller.resident_frame_pool.decode_frame_id_if_present(
                            layer_idx,
                            int(block_idx),
                            kv_head,
                        )
                        if frame_id is None:
                            frame_id = controller.resident_frame_pool.frame_id(
                                layer_idx,
                                int(block_idx),
                                kv_head,
                            )
                        selected_frame_ids[h, j] = frame_id
                last_frame_ids = np.empty((physical_indices.shape[0],), dtype=np.int64)
                for h in range(physical_indices.shape[0]):
                    kv_head = int(kv_head_indices_1d[h])
                    last_block_idx = int(controller.kv_cache.active_indices[-1])
                    last_frame_id = controller.resident_frame_pool.decode_frame_id_if_present(
                        layer_idx,
                        last_block_idx,
                        kv_head,
                    )
                    if last_frame_id is None:
                        last_frame_id = controller.resident_frame_pool.frame_id(
                            layer_idx,
                            last_block_idx,
                            kv_head,
                        )
                    last_frame_ids[h] = last_frame_id
                selected_frames_mx, frames_mx = (
                    controller.resident_frame_pool.get_selected_frame_arena_with_last_by_frame_ids_stable(
                        layer_idx,
                        selected_frame_ids,
                        last_frame_ids,
                        q.dtype,
                    )
                )
                if timing_hook:
                    timing_hook("decode_disk_read", t_disk, frames_mx)
                if timing_hook:
                    t_selected_metal = time.perf_counter()
                out = selected_frame_decode_attention_with_stable_arena(
                    q,
                    selected_frames_mx,
                    frames_mx,
                    num_kv_heads=num_kv_heads,
                    last_len=valid_len,
                    scale=float(1.0 / np.sqrt(q.shape[-1])),
                )
                if timing_hook:
                    timing_hook(
                        "decode_selected_page_metal_attn",
                        t_selected_metal,
                        out,
                    )
                return out
            elif selected_frame_mode == "direct":
                selected_frames_mx, direct_frames = (
                    controller.kv_cache.load_selected_direct_frame_indices_mx(
                        layer_idx,
                        physical_indices,
                        kv_head_indices,
                        q.dtype,
                    )
                )
                max_direct_frames = int(
                    os.environ.get("ALAYAJET_QUEST_DIRECT_FRAME_MAX", "16")
                )
                if len(direct_frames) <= max_direct_frames:
                    last_k_selected = pack_last_page_head_major(last_k_token_mx)
                    last_v_selected = pack_last_page_head_major(last_v_token_mx)
                    if timing_hook:
                        timing_hook("decode_disk_read", t_disk, *direct_frames)
                    if timing_hook:
                        t_selected_metal = time.perf_counter()
                    out = selected_direct_frame_decode_attention(
                        q,
                        selected_frames_mx,
                        direct_frames,
                        last_k_selected,
                        last_v_selected,
                        scale=float(1.0 / np.sqrt(q.shape[-1])),
                        last_layout="head_major",
                    )
                    if timing_hook:
                        timing_hook(
                            "decode_selected_page_metal_attn",
                            t_selected_metal,
                            out,
                        )
                    return out
                selected_frames_mx, frames_mx = _load_selected_frame_pages(
                    controller,
                    layer_idx,
                    physical_indices,
                    kv_head_indices,
                    q.dtype,
                )
            else:
                selected_frames_mx, frames_mx = _load_selected_frame_pages(
                    controller,
                    layer_idx,
                    physical_indices,
                    kv_head_indices,
                    q.dtype,
                )
            last_k_selected = pack_last_page_head_major(last_k_token_mx)
            last_v_selected = pack_last_page_head_major(last_v_token_mx)
            if timing_hook:
                timing_hook("decode_disk_read", t_disk, frames_mx)
            if timing_hook:
                t_selected_metal = time.perf_counter()
            out = selected_frame_decode_attention(
                q,
                selected_frames_mx,
                frames_mx,
                last_k_selected,
                last_v_selected,
                scale=float(1.0 / np.sqrt(q.shape[-1])),
                last_layout="head_major",
            )
            if timing_hook:
                timing_hook("decode_selected_page_metal_attn", t_selected_metal, out)
        elif metal_kv_layout == "head_major":
            selected_pages_mx, kv_pages = _load_selected_kv_head_pages(
                controller,
                layer_idx,
                physical_indices,
                kv_head_indices,
                q.dtype,
            )
            selected_kv_layout = "head_major"
        elif metal_kv_layout == "page_major":
            selected_offsets = np.searchsorted(selected_physical, physical_indices).astype(
                np.int32,
                copy=False,
            )
            selected_pages_mx = mx.array(selected_offsets)
            selected_cache_key = (
                layer_idx,
                metal_kv_layout,
                tuple(int(x) for x in selected_physical.tolist()),
            )
            kv_pages = controller._metal_selected_cache.get(selected_cache_key)
            if kv_pages is None:
                for stale_key in list(controller._metal_selected_cache):
                    if stale_key[0] == layer_idx:
                        del controller._metal_selected_cache[stale_key]
                kv_pages = controller.kv_cache.load_pages_mx(
                    layer_idx,
                    selected_physical.tolist(),
                    dtype=q.dtype,
                )
                controller._metal_selected_cache[selected_cache_key] = kv_pages
            selected_kv_layout = "page_major"
        else:
            raise ValueError(
                "ALAYAJET_QUEST_METAL_KV_LAYOUT must be 'page_major' or 'head_major'"
            )
        if selected_frame_mode == "off":
            if metal_kv_layout == "head_major":
                last_k_selected = pack_last_page_head_major(last_k_token_mx)
                last_v_selected = pack_last_page_head_major(last_v_token_mx)
                last_layout = "head_major"
            else:
                last_k_selected = last_k_token_mx
                last_v_selected = last_v_token_mx
                last_layout = "token_major"
            if timing_hook:
                timing_hook("decode_disk_read", t_disk, kv_pages)
            if timing_hook:
                t_selected_metal = time.perf_counter()
            out = selected_indexed_page_decode_attention(
                q,
                selected_pages_mx,
                kv_pages,
                last_k_selected,
                last_v_selected,
                scale=float(1.0 / np.sqrt(q.shape[-1])),
                kv_layout=selected_kv_layout,
                last_layout=last_layout,
            )
            if timing_hook:
                timing_hook("decode_selected_page_metal_attn", t_selected_metal, out)
    else:
        k_disk_mx, v_disk_mx = controller.kv_cache.load_kv_slices_mx(
            layer_idx,
            physical_indices,
            kv_head_indices,
            dtype=q.dtype,
        )
        if timing_hook:
            timing_hook("decode_disk_read", t_disk)
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
