from typing import Iterable, Optional, Tuple

import mlx.core as mx
import numpy as np
import os
import time
from .kv_cache import QuestController

try:
    import fcntl
except Exception:  # pragma: no cover - optional on some platforms
    fcntl = None


def _maybe_repeat_kv(
    k: mx.array,
    v: mx.array,
    num_heads: int,
    num_kv_heads: int,
) -> Tuple[mx.array, mx.array]:
    if num_kv_heads == num_heads:
        return k, v
    n_rep = num_heads // num_kv_heads
    return mx.repeat(k, n_rep, axis=1), mx.repeat(v, n_rep, axis=1)


def _merge_attn_states(
    prefix_out: mx.array,
    prefix_lse: mx.array,
    suffix_out: mx.array,
    suffix_lse: mx.array,
) -> Tuple[mx.array, mx.array]:
    m = mx.maximum(prefix_lse, suffix_lse)
    exp_prefix = mx.exp(prefix_lse - m)
    exp_suffix = mx.exp(suffix_lse - m)
    denom = exp_prefix + exp_suffix
    out = (prefix_out * exp_prefix + suffix_out * exp_suffix) / denom
    if out.dtype != prefix_out.dtype:
        out = out.astype(prefix_out.dtype)
    lse = m + mx.log(denom)
    return out, lse


def _set_os_nocache(fd: int, enabled: bool) -> None:
    if fcntl is None or not hasattr(fcntl, "F_NOCACHE"):
        return
    try:
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1 if enabled else 0)
    except Exception:
        pass


def _read_pages_into(
    controller: QuestController,
    layer_idx: int,
    page_indices: Iterable[int],
    out_buffer: np.ndarray,
    timing: Optional[list] = None,
    timing_idx: Optional[int] = None,
) -> int:
    page_indices = list(page_indices)
    if not page_indices:
        return 0
    pool = controller.kv_cache.buffer_pool
    page_bytes = pool.page_bytes
    start_time = time.perf_counter() if timing is not None else None
    for i, page_idx in enumerate(page_indices):
        for kv_head in range(controller.num_kv_heads):
            offset = (
                (layer_idx * pool.capacity + page_idx) * pool.num_heads + kv_head
            ) * page_bytes
            data = os.pread(pool.fd, page_bytes, offset)
            if len(data) != page_bytes:
                raise RuntimeError(
                    f"Short read for page {(layer_idx, page_idx, kv_head)}: "
                    f"{len(data)} != {page_bytes}"
                )
            out_buffer[i, :, :, kv_head, :] = np.frombuffer(
                data, dtype=pool.dtype
            ).reshape(pool.frame_shape)
    if timing is not None and timing_idx is not None and start_time is not None:
        timing[timing_idx] = (start_time, time.perf_counter(), len(page_indices))
    return len(page_indices)


def _pages_np_to_kv(
    pages_np: np.ndarray,
    dtype,
) -> Tuple[mx.array, mx.array]:
    if pages_np.size == 0:
        empty = mx.zeros((1, pages_np.shape[3], 0, pages_np.shape[4]), dtype=dtype)
        return empty, empty
    pages_mx = mx.array(pages_np)
    if pages_mx.dtype != dtype:
        pages_mx = pages_mx.astype(dtype)
    k_pages = pages_mx[:, 0]
    v_pages = pages_mx[:, 1]
    k_pages = k_pages.transpose(2, 0, 1, 3).reshape(
        pages_mx.shape[3], -1, pages_mx.shape[4]
    )
    v_pages = v_pages.transpose(2, 0, 1, 3).reshape(
        pages_mx.shape[3], -1, pages_mx.shape[4]
    )
    k_pages = mx.expand_dims(k_pages, axis=0)
    v_pages = mx.expand_dims(v_pages, axis=0)
    return k_pages, v_pages


def _load_prefix_tail(
    controller: QuestController,
    layer_idx: int,
    page_idx: int,
    tail_len: int,
    dtype,
) -> Tuple[mx.array, mx.array]:
    page = None
    active_indices = controller.kv_cache.active_indices
    if active_indices and page_idx == active_indices[-1]:
        page = controller.kv_cache.get_active_buffer_mx(layer_idx)
    if page is None:
        pool = controller.kv_cache.buffer_pool
        page_np = np.empty(
            (1, 2, controller.page_size, controller.num_kv_heads, controller.head_dim),
            dtype=pool.dtype,
        )
        _read_pages_into(controller, layer_idx, [page_idx], page_np)
        page = mx.array(page_np[0])
    if page.dtype != dtype:
        page = page.astype(dtype)
    k_tail = page[0, :tail_len].transpose(1, 0, 2)
    v_tail = page[1, :tail_len].transpose(1, 0, 2)
    k_tail = mx.expand_dims(k_tail, axis=0)
    v_tail = mx.expand_dims(v_tail, axis=0)
    return k_tail, v_tail


def prefill_with_kv_cache_pipelined(
    q_p: mx.array,
    k_p: mx.array,
    v_p: mx.array,
    controller: QuestController,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    chunk_pages: int = 8,
) -> mx.array:
    """
    Prefill attention with chunked prefix KV reads and LSE merge.
    Uses a background prefetch to overlap disk reads with compute.
    """
    prefix_len = controller.kv_cache.seq_len - q_p.shape[2]
    scale = 1.0 / mx.sqrt(head_dim)

    k_self, v_self = _maybe_repeat_kv(k_p, v_p, num_heads, num_kv_heads)
    if prefix_len <= 0:
        return mx.fast.scaled_dot_product_attention(
            q_p,
            k_self,
            v_self,
            scale=scale,
            mask="causal",
        )

    timing_enabled = getattr(controller, "prefill_io_timing", False)
    timing_base = time.perf_counter() if timing_enabled else None
    self_attn_timing = None

    page_size = controller.page_size
    active_indices = controller.kv_cache.active_indices
    full_pages = prefix_len // page_size
    tail_len = prefix_len % page_size

    pool = controller.kv_cache.buffer_pool
    _set_os_nocache(pool.fd, True)

    prefetch_future = None
    page_chunks = None
    buf_idx = 0
    read_timings = None
    attn_timings = None
    if full_pages > 0:
        chunk_pages = max(1, (q_p.shape[2] + page_size - 1) // page_size)
        page_chunks = [
            active_indices[i : min(full_pages, i + chunk_pages)]
            for i in range(0, full_pages, chunk_pages)
        ]

        if timing_enabled:
            read_timings = [None] * len(page_chunks)
            attn_timings = [None] * len(page_chunks)

        buffers = controller.get_prefill_buffers(
            chunk_pages,
            page_size,
            controller.num_kv_heads,
            controller.head_dim,
            controller.kv_cache.dtype,
        )
        prefetch_executor = controller.get_prefill_executor()
        prefetch_future = prefetch_executor.submit(
            _read_pages_into,
            controller,
            layer_idx,
            page_chunks[0],
            buffers[buf_idx],
            read_timings,
            0,
        )

    self_attn_start = time.perf_counter() if timing_enabled else None
    out, lse = mx.fast.scaled_dot_product_attention(
        q_p,
        k_self,
        v_self,
        scale=scale,
        mask="causal",
        return_softmax_lse=True,
    )
    if timing_enabled:
        mx.eval(out, lse)
        self_attn_timing = (self_attn_start, time.perf_counter())
    out_dtype = out.dtype
    if out_dtype != mx.float32:
        out = out.astype(mx.float32)
    if lse.dtype != mx.float32:
        lse = lse.astype(mx.float32)

    if full_pages > 0:
        for idx in range(len(page_chunks)):
            pages_count = prefetch_future.result()
            next_idx = idx + 1
            next_buf = 1 - buf_idx
            if next_idx < len(page_chunks):
                prefetch_future = prefetch_executor.submit(
                    _read_pages_into,
                    controller,
                    layer_idx,
                    page_chunks[next_idx],
                    buffers[next_buf],
                    read_timings,
                    next_idx,
                )
            pages_np = buffers[buf_idx][:pages_count]
            k_chunk, v_chunk = _pages_np_to_kv(pages_np, dtype=q_p.dtype)
            k_chunk, v_chunk = _maybe_repeat_kv(
                k_chunk, v_chunk, num_heads, num_kv_heads
            )
            attn_start = time.perf_counter() if timing_enabled else None
            out_chunk, lse_chunk = mx.fast.scaled_dot_product_attention(
                q_p,
                k_chunk,
                v_chunk,
                scale=scale,
                mask=None,
                return_softmax_lse=True,
            )
            if timing_enabled:
                mx.eval(out_chunk, lse_chunk)
                attn_timings[idx] = (attn_start, time.perf_counter())
            if out_chunk.dtype != out.dtype:
                out_chunk = out_chunk.astype(out.dtype)
            if lse_chunk.dtype != lse.dtype:
                lse_chunk = lse_chunk.astype(lse.dtype)
            out, lse = _merge_attn_states(out, lse, out_chunk, lse_chunk)
            mx.eval(out, lse)
            del pages_np, k_chunk, v_chunk, out_chunk, lse_chunk
            buf_idx = next_buf

    if tail_len > 0:
        tail_page = active_indices[full_pages]
        k_tail, v_tail = _load_prefix_tail(
            controller, layer_idx, tail_page, tail_len, dtype=q_p.dtype
        )
        k_tail, v_tail = _maybe_repeat_kv(k_tail, v_tail, num_heads, num_kv_heads)
        tail_attn_start = time.perf_counter() if timing_enabled else None
        out_tail, lse_tail = mx.fast.scaled_dot_product_attention(
            q_p,
            k_tail,
            v_tail,
            scale=scale,
            mask=None,
            return_softmax_lse=True,
        )
        if timing_enabled:
            mx.eval(out_tail, lse_tail)
            tail_attn_end = time.perf_counter()
            base = timing_base or tail_attn_end
            print(
                "[Quest][Prefill][IO] tail_attn "
                f"| {((tail_attn_end - tail_attn_start) * 1000.0):.2f}ms "
                f"| t={((tail_attn_start - base) * 1000.0):.2f}-"
                f"{((tail_attn_end - base) * 1000.0):.2f}ms"
            )
        if out_tail.dtype != out.dtype:
            out_tail = out_tail.astype(out.dtype)
        if lse_tail.dtype != lse.dtype:
            lse_tail = lse_tail.astype(lse.dtype)
        out, lse = _merge_attn_states(out, lse, out_tail, lse_tail)

    if timing_enabled:
        base = timing_base or time.perf_counter()
        if self_attn_timing is not None:
            self_start, self_end = self_attn_timing
            print(
                "[Quest][Prefill][IO] self_attn "
                f"| {((self_end - self_start) * 1000.0):.2f}ms "
                f"| t={((self_start - base) * 1000.0):.2f}-"
                f"{((self_end - base) * 1000.0):.2f}ms"
            )
        if read_timings:
            total_read = 0.0
            total_overlap = 0.0
            # read0 overlaps with self-attn
            if self_attn_timing is not None and read_timings[0] is not None:
                read_start, read_end, pages_count = read_timings[0]
                self_start, self_end = self_attn_timing
                overlap = max(0.0, min(read_end, self_end) - max(read_start, self_start))
                read_dur = read_end - read_start
                total_read += read_dur
                total_overlap += overlap
                coverage = overlap / read_dur if read_dur > 0 else 0.0
                read_start_ms = (read_start - base) * 1000.0
                read_end_ms = (read_end - base) * 1000.0
                self_start_ms = (self_start - base) * 1000.0
                self_end_ms = (self_end - base) * 1000.0
                tokens = pages_count * page_size
                print(
                    "[Quest][Prefill][IO] read 00 vs self_attn "
                    f"| read {read_dur * 1000.0:.2f}ms "
                    f"(t={read_start_ms:.2f}-{read_end_ms:.2f}ms, {tokens} tok) "
                    f"| attn {((self_end - self_start) * 1000.0):.2f}ms "
                    f"(t={self_start_ms:.2f}-{self_end_ms:.2f}ms) "
                    f"| overlap {overlap * 1000.0:.2f}ms "
                    f"({coverage * 100.0:.1f}%)"
                )
            # read i overlaps with attn (i-1)
            if attn_timings:
                for idx in range(1, len(read_timings)):
                    read_t = read_timings[idx]
                    attn_t = attn_timings[idx - 1] if idx - 1 < len(attn_timings) else None
                    if read_t is None or attn_t is None:
                        continue
                    read_start, read_end, pages_count = read_t
                    attn_start, attn_end = attn_t
                    overlap = max(
                        0.0, min(read_end, attn_end) - max(read_start, attn_start)
                    )
                    read_dur = read_end - read_start
                    attn_dur = attn_end - attn_start
                    total_read += read_dur
                    total_overlap += overlap
                    coverage = overlap / read_dur if read_dur > 0 else 0.0
                    read_start_ms = (read_start - base) * 1000.0
                    read_end_ms = (read_end - base) * 1000.0
                    attn_start_ms = (attn_start - base) * 1000.0
                    attn_end_ms = (attn_end - base) * 1000.0
                    tokens = pages_count * page_size
                    print(
                        "[Quest][Prefill][IO] read "
                        f"{idx:02d} vs attn {idx - 1:02d} "
                        f"| read {read_dur * 1000.0:.2f}ms "
                        f"(t={read_start_ms:.2f}-{read_end_ms:.2f}ms, {tokens} tok) "
                        f"| attn {attn_dur * 1000.0:.2f}ms "
                        f"(t={attn_start_ms:.2f}-{attn_end_ms:.2f}ms) "
                        f"| overlap {overlap * 1000.0:.2f}ms "
                        f"({coverage * 100.0:.1f}%)"
                    )
            if total_read > 0:
                overall = total_overlap / total_read * 100.0
                print(
                    "[Quest][Prefill][IO] overlap_total "
                    f"{overall:.1f}% (read={total_read*1000.0:.2f}ms, "
                    f"overlap={total_overlap*1000.0:.2f}ms)"
                )

    if out.dtype != out_dtype:
        out = out.astype(out_dtype)
    return out
