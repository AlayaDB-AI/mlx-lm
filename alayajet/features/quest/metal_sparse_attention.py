from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


_FUSED_SPARSE_DECODE_SOURCE = r"""
uint h = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint nt = threads_per_threadgroup.x;
if (h >= uint(num_heads)) {
    return;
}

const int kv_h = int(h) / GROUP_SIZE;
const int selected_count = page_budget < num_pages ? page_budget : num_pages;
const int total_tokens = selected_count * PAGE_SIZE + last_len;

threadgroup float page_scores[MAX_PAGES];
threadgroup int selected[MAX_BUDGET];
threadgroup float token_scores[MAX_TOKENS];
threadgroup float reduce_buf[THREADGROUP_SIZE];
threadgroup float denom_buf[1];

for (uint page = tid; page < uint(MAX_PAGES); page += nt) {
    page_scores[page] = -INFINITY;
}
for (uint i = tid; i < uint(MAX_BUDGET); i += nt) {
    selected[i] = -1;
}
for (uint i = tid; i < uint(MAX_TOKENS); i += nt) {
    token_scores[i] = -INFINITY;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint page_u = tid; page_u < uint(num_pages); page_u += nt) {
    const int page = int(page_u);
    float score = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
        const float qv = float(q[h * HEAD_DIM + d]);
        const int meta_idx = (kv_h * num_pages + page) * HEAD_DIM + d;
        const float kval = qv >= 0.0f ? float(k_max[meta_idx]) : float(k_min[meta_idx]);
        score += qv * kval;
    }
    page_scores[page] = score;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid == 0) {
    for (int rank = 0; rank < selected_count; ++rank) {
        float best_score = -INFINITY;
        int best_page = -1;

        for (int page = 0; page < num_pages; ++page) {
            const float score = page_scores[page];
            if (score > best_score) {
                best_score = score;
                best_page = page;
            }
        }
        selected[rank] = best_page;
        if (best_page >= 0) {
            page_scores[best_page] = -INFINITY;
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float local_max = -INFINITY;
for (uint token_u = tid; token_u < uint(total_tokens); token_u += nt) {
    const int token = int(token_u);
    bool in_last = token >= selected_count * PAGE_SIZE;
    int page = 0;
    int page_offset = 0;
    if (in_last) {
        page_offset = token - selected_count * PAGE_SIZE;
    } else {
        const int rank = token / PAGE_SIZE;
        page = selected[rank];
        page_offset = token - rank * PAGE_SIZE;
    }

    float score = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
        const float qv = float(q[h * HEAD_DIM + d]);
        if (in_last) {
            const int k_idx = (page_offset * num_kv_heads + kv_h) * HEAD_DIM + d;
            score += qv * float(last_k[k_idx]);
        } else {
            const int k_idx = (((page * 2) * PAGE_SIZE + page_offset) * num_kv_heads + kv_h) * HEAD_DIM + d;
            score += qv * float(kv_pages[k_idx]);
        }
    }
    score *= scale;
    token_scores[token] = score;
    local_max = metal::max(local_max, score);
}
reduce_buf[tid] = local_max;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = nt >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
        reduce_buf[tid] = metal::max(reduce_buf[tid], reduce_buf[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
const float max_score = reduce_buf[0];

float local_denom = 0.0f;
for (uint token_u = tid; token_u < uint(total_tokens); token_u += nt) {
    local_denom += metal::precise::exp(token_scores[token_u] - max_score);
}
reduce_buf[tid] = local_denom;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = nt >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
        reduce_buf[tid] += reduce_buf[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (tid == 0) {
    denom_buf[0] = reduce_buf[0];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float denom = denom_buf[0];

for (uint token_u = tid; token_u < uint(total_tokens); token_u += nt) {
    token_scores[token_u] = denom > 0.0f
        ? metal::precise::exp(token_scores[token_u] - max_score) / denom
        : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const uint d_lane = tid / uint(HEAD_DIM);
const int d = int(tid % uint(HEAD_DIM));
float local_acc = 0.0f;
for (int token = int(d_lane); token < total_tokens; token += LANES_PER_DIM) {
    const float weight = token_scores[token];
    bool in_last = token >= selected_count * PAGE_SIZE;
    int page_offset = 0;
    float vv = 0.0f;
    if (in_last) {
        page_offset = token - selected_count * PAGE_SIZE;
        const int v_idx = (page_offset * num_kv_heads + kv_h) * HEAD_DIM + d;
        vv = float(last_v[v_idx]);
    } else {
        const int rank = token / PAGE_SIZE;
        const int page = selected[rank];
        page_offset = token - rank * PAGE_SIZE;
        const int v_idx = (((page * 2 + 1) * PAGE_SIZE + page_offset) * num_kv_heads + kv_h) * HEAD_DIM + d;
        vv = float(kv_pages[v_idx]);
    }
    local_acc += weight * vv;
}
reduce_buf[tid] = local_acc;
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid < uint(HEAD_DIM)) {
    float acc = 0.0f;
    for (int lane = 0; lane < LANES_PER_DIM; ++lane) {
        acc += reduce_buf[lane * HEAD_DIM + d];
    }
    out[h * HEAD_DIM + d] = T(acc);
}
"""


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


@lru_cache(maxsize=64)
def _fused_sparse_decode_kernel(
    head_dim: int,
    page_size: int,
    max_budget: int,
    group_size: int,
    max_pages: int,
    threadgroup_size: int,
):
    if head_dim not in (64, 128):
        raise ValueError(f"Metal sparse decode only supports head_dim 64/128, got {head_dim}")
    if max_budget <= 0:
        raise ValueError("max_budget must be positive")
    return mx.fast.metal_kernel(
        name=(
            f"alayajet_fused_sparse_decode_d{head_dim}_p{page_size}"
            f"_b{max_budget}_g{group_size}_n{max_pages}_t{threadgroup_size}"
        ),
        input_names=[
            "q",
            "k_min",
            "k_max",
            "kv_pages",
            "last_k",
            "last_v",
            "num_heads",
            "num_kv_heads",
            "num_pages",
            "page_budget",
            "last_len",
            "scale",
        ],
        output_names=["out"],
        source=_FUSED_SPARSE_DECODE_SOURCE,
        ensure_row_contiguous=True,
    )


def fused_sparse_decode_attention(
    q: mx.array,
    k_min: mx.array,
    k_max: mx.array,
    kv_pages: mx.array,
    last_k: mx.array,
    last_v: mx.array,
    *,
    page_budget: int,
    scale: float,
) -> mx.array:
    """Fused Metal decode attention over paged KV candidates.

    Inputs:
      q: (1, Hq, 1, D) or (Hq, D)
      k_min/k_max: (Hkv, N, D), metadata aligned with ``kv_pages``.
      kv_pages: (N, 2, page_size, Hkv, D)
      last_k/last_v: (last_len, Hkv, D)

    Returns:
      (1, Hq, 1, D), matching ``mx.fast.scaled_dot_product_attention`` decode output.
    """
    if q.ndim == 4:
        q_2d = q[0, :, 0, :]
    elif q.ndim == 2:
        q_2d = q
    else:
        raise ValueError(f"Expected q rank 2 or 4, got shape {q.shape}")

    num_heads, head_dim = q_2d.shape
    num_kv_heads = k_min.shape[0]
    num_pages = k_min.shape[1]
    page_size = kv_pages.shape[2] if num_pages else 1
    last_len = last_k.shape[0]
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    if page_budget <= 0:
        raise ValueError("page_budget must be positive")

    group_size = num_heads // num_kv_heads
    max_budget = max(1, int(page_budget))
    max_pages = _next_power_of_two(max(1, int(num_pages)))
    threadgroup_size = 256
    kernel = _fused_sparse_decode_kernel(
        head_dim,
        page_size,
        max_budget,
        group_size,
        max_pages,
        threadgroup_size,
    )
    out = kernel(
        inputs=[
            q_2d,
            k_min,
            k_max,
            kv_pages,
            last_k,
            last_v,
            num_heads,
            num_kv_heads,
            num_pages,
            min(int(page_budget), int(num_pages)),
            last_len,
            float(scale),
        ],
        template=[
            ("T", q_2d.dtype),
            ("HEAD_DIM", int(head_dim)),
            ("PAGE_SIZE", int(page_size)),
            ("MAX_BUDGET", int(max_budget)),
            ("GROUP_SIZE", int(group_size)),
            ("MAX_PAGES", int(max_pages)),
            ("MAX_TOKENS", int(max_budget * page_size + page_size)),
            ("THREADGROUP_SIZE", int(threadgroup_size)),
            ("LANES_PER_DIM", int(threadgroup_size // head_dim)),
        ],
        grid=(int(num_heads * threadgroup_size), 1, 1),
        threadgroup=(int(threadgroup_size), 1, 1),
        output_shapes=[(int(num_heads), int(head_dim))],
        output_dtypes=[q_2d.dtype],
        stream=mx.gpu,
    )[0]
    return out.reshape(1, num_heads, 1, head_dim)
