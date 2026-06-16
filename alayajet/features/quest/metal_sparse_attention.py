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
    for (int i = 1; i < selected_count; ++i) {
        const int key = selected[i];
        int j = i - 1;
        while (j >= 0 && selected[j] > key) {
            selected[j + 1] = selected[j];
            --j;
        }
        selected[j + 1] = key;
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
            const int k_idx = LAST_HEAD_MAJOR != 0
                ? (kv_h * last_len + page_offset) * HEAD_DIM + d
                : (page_offset * num_kv_heads + kv_h) * HEAD_DIM + d;
            score += qv * float(last_k[k_idx]);
        } else {
            const int k_idx = KV_HEAD_MAJOR != 0
                ? (((kv_h * num_pages + page) * 2 * PAGE_SIZE) + page_offset) * HEAD_DIM + d
                : (((page * 2) * PAGE_SIZE + page_offset) * num_kv_heads + kv_h) * HEAD_DIM + d;
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
        const int v_idx = LAST_HEAD_MAJOR != 0
            ? (kv_h * last_len + page_offset) * HEAD_DIM + d
            : (page_offset * num_kv_heads + kv_h) * HEAD_DIM + d;
        vv = float(last_v[v_idx]);
    } else {
        const int rank = token / PAGE_SIZE;
        const int page = selected[rank];
        page_offset = token - rank * PAGE_SIZE;
        const int v_idx = KV_HEAD_MAJOR != 0
            ? (((kv_h * num_pages + page) * 2 * PAGE_SIZE) + PAGE_SIZE + page_offset) * HEAD_DIM + d
            : (((page * 2 + 1) * PAGE_SIZE + page_offset) * num_kv_heads + kv_h) * HEAD_DIM + d;
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


_SELECTED_PAGE_DECODE_SOURCE = r"""
uint h = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint nt = threads_per_threadgroup.x;
if (h >= uint(num_heads)) {
    return;
}

const int total_tokens = selected_tokens + last_len;

threadgroup float token_scores[MAX_TOKENS];
threadgroup float reduce_buf[THREADGROUP_SIZE];
threadgroup float denom_buf[1];

for (uint i = tid; i < uint(MAX_TOKENS); i += nt) {
    token_scores[i] = -INFINITY;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float local_max = -INFINITY;
for (uint token_u = tid; token_u < uint(total_tokens); token_u += nt) {
    const int token = int(token_u);
    const bool in_last = token >= selected_tokens;
    const int token_offset = in_last ? token - selected_tokens : token;

    float score = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
        const float qv = float(q[h * HEAD_DIM + d]);
        const int k_idx = in_last
            ? (int(h) * last_len + token_offset) * HEAD_DIM + d
            : (int(h) * selected_tokens + token_offset) * HEAD_DIM + d;
        score += qv * float(in_last ? last_k[k_idx] : selected_k[k_idx]);
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
    const bool in_last = token >= selected_tokens;
    const int token_offset = in_last ? token - selected_tokens : token;
    const int v_idx = in_last
        ? (int(h) * last_len + token_offset) * HEAD_DIM + d
        : (int(h) * selected_tokens + token_offset) * HEAD_DIM + d;
    const float vv = float(in_last ? last_v[v_idx] : selected_v[v_idx]);
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


_SELECTED_INDEXED_PAGE_DECODE_SOURCE = r"""
uint h = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint nt = threads_per_threadgroup.x;
if (h >= uint(num_heads)) {
    return;
}

const int kv_h = int(h) / GROUP_SIZE;
const int total_tokens = selected_count * PAGE_SIZE + last_len;

threadgroup float token_scores[MAX_TOKENS];
threadgroup float reduce_buf[THREADGROUP_SIZE];
threadgroup float denom_buf[1];

for (uint i = tid; i < uint(MAX_TOKENS); i += nt) {
    token_scores[i] = -INFINITY;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float local_max = -INFINITY;
for (uint token_u = tid; token_u < uint(total_tokens); token_u += nt) {
    const int token = int(token_u);
    const bool in_last = token >= selected_count * PAGE_SIZE;
    int page_offset = 0;
    int page = 0;
    if (in_last) {
        page_offset = token - selected_count * PAGE_SIZE;
    } else {
        const int rank = token / PAGE_SIZE;
        page = int(selected_pages[int(h) * selected_count + rank]);
        page_offset = token - rank * PAGE_SIZE;
    }

    float score = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
        const float qv = float(q[h * HEAD_DIM + d]);
        if (in_last) {
            const int k_idx = LAST_HEAD_MAJOR != 0
                ? (kv_h * last_len + page_offset) * HEAD_DIM + d
                : (page_offset * num_kv_heads + kv_h) * HEAD_DIM + d;
            score += qv * float(last_k[k_idx]);
        } else {
            const int k_idx = KV_HEAD_MAJOR != 0
                ? (((kv_h * num_pages + page) * 2 * PAGE_SIZE) + page_offset) * HEAD_DIM + d
                : (((page * 2) * PAGE_SIZE + page_offset) * num_kv_heads + kv_h) * HEAD_DIM + d;
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
    const bool in_last = token >= selected_count * PAGE_SIZE;
    int page_offset = 0;
    float vv = 0.0f;
    if (in_last) {
        page_offset = token - selected_count * PAGE_SIZE;
        const int v_idx = LAST_HEAD_MAJOR != 0
            ? (kv_h * last_len + page_offset) * HEAD_DIM + d
            : (page_offset * num_kv_heads + kv_h) * HEAD_DIM + d;
        vv = float(last_v[v_idx]);
    } else {
        const int rank = token / PAGE_SIZE;
        const int page = int(selected_pages[int(h) * selected_count + rank]);
        page_offset = token - rank * PAGE_SIZE;
        const int v_idx = KV_HEAD_MAJOR != 0
            ? (((kv_h * num_pages + page) * 2 * PAGE_SIZE) + PAGE_SIZE + page_offset) * HEAD_DIM + d
            : (((page * 2 + 1) * PAGE_SIZE + page_offset) * num_kv_heads + kv_h) * HEAD_DIM + d;
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


_SELECTED_FRAME_DECODE_SOURCE = r"""
uint h = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint nt = threads_per_threadgroup.x;
if (h >= uint(num_heads)) {
    return;
}

const int kv_h = int(h) / GROUP_SIZE;
const int total_tokens = selected_count * PAGE_SIZE + last_len;

threadgroup float token_scores[MAX_TOKENS];
threadgroup float reduce_buf[THREADGROUP_SIZE];
threadgroup float denom_buf[1];

for (uint i = tid; i < uint(MAX_TOKENS); i += nt) {
    token_scores[i] = -INFINITY;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float local_max = -INFINITY;
for (uint token_u = tid; token_u < uint(total_tokens); token_u += nt) {
    const int token = int(token_u);
    const bool in_last = token >= selected_count * PAGE_SIZE;
    int page_offset = 0;
    int frame = 0;
    if (in_last) {
        page_offset = token - selected_count * PAGE_SIZE;
    } else {
        const int rank = token / PAGE_SIZE;
        frame = int(selected_frames[int(h) * selected_count + rank]);
        page_offset = token - rank * PAGE_SIZE;
    }

    float score = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
        const float qv = float(q[h * HEAD_DIM + d]);
        if (in_last) {
            const int k_idx = LAST_HEAD_MAJOR != 0
                ? (kv_h * last_len + page_offset) * HEAD_DIM + d
                : (page_offset * num_kv_heads + kv_h) * HEAD_DIM + d;
            score += qv * float(last_k[k_idx]);
        } else {
            const int k_idx = (((frame * 2 * PAGE_SIZE) + page_offset) * HEAD_DIM) + d;
            score += qv * float(frames[k_idx]);
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
    const bool in_last = token >= selected_count * PAGE_SIZE;
    int page_offset = 0;
    float vv = 0.0f;
    if (in_last) {
        page_offset = token - selected_count * PAGE_SIZE;
        const int v_idx = LAST_HEAD_MAJOR != 0
            ? (kv_h * last_len + page_offset) * HEAD_DIM + d
            : (page_offset * num_kv_heads + kv_h) * HEAD_DIM + d;
        vv = float(last_v[v_idx]);
    } else {
        const int rank = token / PAGE_SIZE;
        const int frame = int(selected_frames[int(h) * selected_count + rank]);
        page_offset = token - rank * PAGE_SIZE;
        const int v_idx = (((frame * 2 * PAGE_SIZE) + PAGE_SIZE + page_offset) * HEAD_DIM) + d;
        vv = float(frames[v_idx]);
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


_SELECTED_DIRECT_FRAME_DECODE_TEMPLATE = r"""
uint h = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint nt = threads_per_threadgroup.x;
if (h >= uint(num_heads)) {
    return;
}

const int kv_h = int(h) / GROUP_SIZE;
const int total_tokens = selected_count * PAGE_SIZE + last_len;

threadgroup float token_scores[MAX_TOKENS];
threadgroup float reduce_buf[THREADGROUP_SIZE];
threadgroup float denom_buf[1];

for (uint i = tid; i < uint(MAX_TOKENS); i += nt) {
    token_scores[i] = -INFINITY;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float local_max = -INFINITY;
for (uint token_u = tid; token_u < uint(total_tokens); token_u += nt) {
    const int token = int(token_u);
    const bool in_last = token >= selected_count * PAGE_SIZE;
    int page_offset = 0;
    int frame = 0;
    if (in_last) {
        page_offset = token - selected_count * PAGE_SIZE;
    } else {
        const int rank = token / PAGE_SIZE;
        frame = int(selected_frames[int(h) * selected_count + rank]);
        page_offset = token - rank * PAGE_SIZE;
    }

    float score = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
        const float qv = float(q[h * HEAD_DIM + d]);
        if (in_last) {
            const int k_idx = LAST_HEAD_MAJOR != 0
                ? (kv_h * last_len + page_offset) * HEAD_DIM + d
                : (page_offset * num_kv_heads + kv_h) * HEAD_DIM + d;
            score += qv * float(last_k[k_idx]);
        } else {
            float kv = 0.0f;
            const int frame_offset = page_offset * HEAD_DIM + d;
__K_SWITCH__
            score += qv * kv;
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
    const bool in_last = token >= selected_count * PAGE_SIZE;
    int page_offset = 0;
    float vv = 0.0f;
    if (in_last) {
        page_offset = token - selected_count * PAGE_SIZE;
        const int v_idx = LAST_HEAD_MAJOR != 0
            ? (kv_h * last_len + page_offset) * HEAD_DIM + d
            : (page_offset * num_kv_heads + kv_h) * HEAD_DIM + d;
        vv = float(last_v[v_idx]);
    } else {
        const int rank = token / PAGE_SIZE;
        const int frame = int(selected_frames[int(h) * selected_count + rank]);
        page_offset = token - rank * PAGE_SIZE;
        const int frame_offset = (PAGE_SIZE + page_offset) * HEAD_DIM + d;
__V_SWITCH__
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


_SELECTED_FRAME_WITH_LAST_SOURCE = r"""
uint h = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint nt = threads_per_threadgroup.x;
if (h >= uint(num_heads)) {
    return;
}

const int history_count = selected_count - 1;
const int total_tokens = history_count * PAGE_SIZE + last_len;

threadgroup float token_scores[MAX_TOKENS];
threadgroup float reduce_buf[THREADGROUP_SIZE];
threadgroup float denom_buf[1];

for (uint i = tid; i < uint(MAX_TOKENS); i += nt) {
    token_scores[i] = -INFINITY;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float local_max = -INFINITY;
for (uint token_u = tid; token_u < uint(total_tokens); token_u += nt) {
    const int token = int(token_u);
    const bool in_last = token >= history_count * PAGE_SIZE;
    const int rank = in_last ? history_count : token / PAGE_SIZE;
    const int page_offset = in_last ? token - history_count * PAGE_SIZE : token - rank * PAGE_SIZE;
    const int frame = int(selected_frames[int(h) * selected_count + rank]);

    float score = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
        const int k_idx = (((frame * 2 * PAGE_SIZE) + page_offset) * HEAD_DIM) + d;
        score += float(q[h * HEAD_DIM + d]) * float(frames[k_idx]);
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
    const bool in_last = token >= history_count * PAGE_SIZE;
    const int rank = in_last ? history_count : token / PAGE_SIZE;
    const int page_offset = in_last ? token - history_count * PAGE_SIZE : token - rank * PAGE_SIZE;
    const int frame = int(selected_frames[int(h) * selected_count + rank]);
    const int v_idx = (((frame * 2 * PAGE_SIZE) + PAGE_SIZE + page_offset) * HEAD_DIM) + d;
    local_acc += token_scores[token] * float(frames[v_idx]);
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


def _selected_direct_frame_source(max_direct_frames: int) -> str:
    k_lines = []
    v_lines = []
    for idx in range(max_direct_frames):
        prefix = "if" if idx == 0 else "else if"
        k_lines.append(
            f"            {prefix} (frame == {idx}) {{ kv = float(frame_{idx}[frame_offset]); }}"
        )
        v_lines.append(
            f"        {prefix} (frame == {idx}) {{ vv = float(frame_{idx}[frame_offset]); }}"
        )
    return (
        _SELECTED_DIRECT_FRAME_DECODE_TEMPLATE
        .replace("__K_SWITCH__", "\n".join(k_lines))
        .replace("__V_SWITCH__", "\n".join(v_lines))
    )


_SELECTED_DIRECT_FRAME_WITH_LAST_SOURCE_TEMPLATE = r"""
uint h = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint nt = threads_per_threadgroup.x;
if (h >= uint(num_heads)) {
    return;
}

const int history_count = selected_count - 1;
const int total_tokens = history_count * PAGE_SIZE + last_len;

threadgroup float token_scores[MAX_TOKENS];
threadgroup float reduce_buf[THREADGROUP_SIZE];
threadgroup float denom_buf[1];

for (uint i = tid; i < uint(MAX_TOKENS); i += nt) {
    token_scores[i] = -INFINITY;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float local_max = -INFINITY;
for (uint token_u = tid; token_u < uint(total_tokens); token_u += nt) {
    const int token = int(token_u);
    const bool in_last = token >= history_count * PAGE_SIZE;
    const int rank = in_last ? history_count : token / PAGE_SIZE;
    const int page_offset = in_last ? token - history_count * PAGE_SIZE : token - rank * PAGE_SIZE;
    const int frame = int(selected_frames[int(h) * selected_count + rank]);
    const int frame_offset = page_offset * HEAD_DIM;

    float score = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
        float kv = 0.0f;
        const int offset = frame_offset + d;
__K_SWITCH__
        score += float(q[h * HEAD_DIM + d]) * kv;
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
    const bool in_last = token >= history_count * PAGE_SIZE;
    const int rank = in_last ? history_count : token / PAGE_SIZE;
    const int page_offset = in_last ? token - history_count * PAGE_SIZE : token - rank * PAGE_SIZE;
    const int frame = int(selected_frames[int(h) * selected_count + rank]);
    const int offset = (PAGE_SIZE + page_offset) * HEAD_DIM + d;
    float vv = 0.0f;
__V_SWITCH__
    local_acc += token_scores[token] * vv;
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


def _selected_direct_frame_with_last_source(max_direct_frames: int) -> str:
    k_lines = []
    v_lines = []
    for idx in range(max_direct_frames):
        prefix = "if" if idx == 0 else "else if"
        k_lines.append(
            f"        {prefix} (frame == {idx}) {{ kv = float(frame_{idx}[offset]); }}"
        )
        v_lines.append(
            f"    {prefix} (frame == {idx}) {{ vv = float(frame_{idx}[offset]); }}"
        )
    return (
        _SELECTED_DIRECT_FRAME_WITH_LAST_SOURCE_TEMPLATE
        .replace("__K_SWITCH__", "\n".join(k_lines))
        .replace("__V_SWITCH__", "\n".join(v_lines))
    )


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def pack_kv_pages_head_major(kv_pages: mx.array) -> mx.array:
    """Pack KV pages for per-KV-head contiguous Metal decode access.

    Input layout is the Quest disk/cache layout:
      (num_pages, 2, page_size, num_kv_heads, head_dim)

    Output layout is:
      (num_kv_heads, num_pages, 2, page_size, head_dim)

    The fused Metal kernel launches one threadgroup per query head. During
    decode each threadgroup repeatedly scans pages for a fixed KV head, so this
    layout turns token/head-dim reads into contiguous memory instead of striding
    by ``num_kv_heads * head_dim`` for every token.
    """
    if kv_pages.ndim != 5:
        raise ValueError(f"Expected kv_pages rank 5, got shape {kv_pages.shape}")
    return kv_pages.transpose(3, 0, 1, 2, 4)


def pack_last_page_head_major(page: mx.array) -> mx.array:
    """Pack a last-page K/V tensor from (len, Hkv, D) to (Hkv, len, D)."""
    if page.ndim != 3:
        raise ValueError(f"Expected last-page rank 3, got shape {page.shape}")
    return page.transpose(1, 0, 2)


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


@lru_cache(maxsize=64)
def _selected_page_decode_kernel(
    head_dim: int,
    max_tokens: int,
    threadgroup_size: int,
):
    if head_dim not in (64, 128):
        raise ValueError(
            f"Metal selected-page decode only supports head_dim 64/128, got {head_dim}"
        )
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    return mx.fast.metal_kernel(
        name=(
            f"alayajet_selected_page_decode_d{head_dim}"
            f"_n{max_tokens}_t{threadgroup_size}"
        ),
        input_names=[
            "q",
            "selected_k",
            "selected_v",
            "last_k",
            "last_v",
            "num_heads",
            "selected_tokens",
            "last_len",
            "scale",
        ],
        output_names=["out"],
        source=_SELECTED_PAGE_DECODE_SOURCE,
        ensure_row_contiguous=True,
    )


def selected_page_decode_attention(
    q: mx.array,
    selected_k: mx.array,
    selected_v: mx.array,
    last_k: mx.array,
    last_v: mx.array,
    *,
    scale: float,
    max_last_tokens: int,
) -> mx.array:
    """Metal decode attention over already-selected pages and active last page.

    Inputs are per-query-head tensors:
      q: (1, H, 1, D) or (H, D)
      selected_k/selected_v: (H, selected_tokens, D)
      last_k/last_v: (H, last_len, D)

    The kernel deliberately performs no candidate scoring or top-k. It is the
    selected-page-only decode path used after metadata-based page selection.
    """
    if q.ndim == 4:
        q_2d = q[0, :, 0, :]
    elif q.ndim == 2:
        q_2d = q
    else:
        raise ValueError(f"Expected q rank 2 or 4, got shape {q.shape}")

    num_heads, head_dim = q_2d.shape
    if selected_k.ndim != 3 or selected_v.ndim != 3:
        raise ValueError(
            "selected_k/selected_v must have shape (num_heads, selected_tokens, head_dim)"
        )
    if last_k.ndim != 3 or last_v.ndim != 3:
        raise ValueError("last_k/last_v must have shape (num_heads, last_len, head_dim)")
    selected_tokens = int(selected_k.shape[1])
    last_len = int(last_k.shape[1])
    if selected_k.shape != (num_heads, selected_tokens, head_dim):
        raise ValueError(
            "selected_k must have shape "
            f"({num_heads}, {selected_tokens}, {head_dim}), got {selected_k.shape}"
        )
    if selected_v.shape != (num_heads, selected_tokens, head_dim):
        raise ValueError(
            "selected_v must have shape "
            f"({num_heads}, {selected_tokens}, {head_dim}), got {selected_v.shape}"
        )
    if last_k.shape != (num_heads, last_len, head_dim):
        raise ValueError(
            f"last_k must have shape ({num_heads}, {last_len}, {head_dim}), got {last_k.shape}"
        )
    if last_v.shape != (num_heads, last_len, head_dim):
        raise ValueError(
            f"last_v must have shape ({num_heads}, {last_len}, {head_dim}), got {last_v.shape}"
        )
    if selected_tokens == 0 and last_len == 0:
        return mx.zeros((1, num_heads, 1, head_dim), dtype=q_2d.dtype)

    max_tokens = int(selected_tokens + max(1, max_last_tokens))
    if selected_tokens + last_len > max_tokens:
        raise ValueError(
            f"selected_tokens + last_len exceeds max_tokens: "
            f"{selected_tokens} + {last_len} > {max_tokens}"
        )
    threadgroup_size = 256
    kernel = _selected_page_decode_kernel(head_dim, max_tokens, threadgroup_size)
    out = kernel(
        inputs=[
            q_2d,
            selected_k,
            selected_v,
            last_k,
            last_v,
            num_heads,
            selected_tokens,
            last_len,
            float(scale),
        ],
        template=[
            ("T", q_2d.dtype),
            ("HEAD_DIM", int(head_dim)),
            ("MAX_TOKENS", int(max_tokens)),
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


@lru_cache(maxsize=64)
def _selected_indexed_page_decode_kernel(
    head_dim: int,
    page_size: int,
    max_budget: int,
    group_size: int,
    max_pages: int,
    threadgroup_size: int,
):
    if head_dim not in (64, 128):
        raise ValueError(
            f"Metal selected-page decode only supports head_dim 64/128, got {head_dim}"
        )
    if max_budget <= 0:
        raise ValueError("max_budget must be positive")
    return mx.fast.metal_kernel(
        name=(
            f"alayajet_selected_indexed_page_decode_d{head_dim}_p{page_size}"
            f"_b{max_budget}_g{group_size}_n{max_pages}_t{threadgroup_size}"
        ),
        input_names=[
            "q",
            "selected_pages",
            "kv_pages",
            "last_k",
            "last_v",
            "num_heads",
            "num_kv_heads",
            "num_pages",
            "selected_count",
            "last_len",
            "scale",
        ],
        output_names=["out"],
        source=_SELECTED_INDEXED_PAGE_DECODE_SOURCE,
        ensure_row_contiguous=True,
    )


def selected_indexed_page_decode_attention(
    q: mx.array,
    selected_pages: mx.array,
    kv_pages: mx.array,
    last_k: mx.array,
    last_v: mx.array,
    *,
    scale: float,
    kv_layout: str = "page_major",
    last_layout: str = "token_major",
) -> mx.array:
    """Metal decode attention over a selected-page union plus per-head indices.

    ``selected_pages`` is shaped (Hq, selected_count) and indexes into
    ``kv_pages``. Unlike ``selected_page_decode_attention``, this keeps K/V in
    KV-head layout and avoids materializing one selected K/V tensor per query
    head, which is important for GQA/MQA models.
    """
    if q.ndim == 4:
        q_2d = q[0, :, 0, :]
    elif q.ndim == 2:
        q_2d = q
    else:
        raise ValueError(f"Expected q rank 2 or 4, got shape {q.shape}")

    num_heads, head_dim = q_2d.shape
    if selected_pages.ndim != 2 or selected_pages.shape[0] != num_heads:
        raise ValueError(
            f"selected_pages must have shape ({num_heads}, selected_count), "
            f"got {selected_pages.shape}"
        )
    selected_count = int(selected_pages.shape[1])
    if kv_layout == "page_major":
        num_pages = int(kv_pages.shape[0])
        page_size = int(kv_pages.shape[2]) if num_pages else 1
        num_kv_heads = int(kv_pages.shape[3]) if num_pages else int(num_heads)
        kv_head_major = False
    elif kv_layout == "head_major":
        num_kv_heads = int(kv_pages.shape[0])
        num_pages = int(kv_pages.shape[1])
        page_size = int(kv_pages.shape[3]) if num_pages else 1
        kv_head_major = True
    else:
        raise ValueError(f"Unsupported kv_layout: {kv_layout}")
    if last_layout == "token_major":
        last_len = int(last_k.shape[0])
        last_head_major = False
    elif last_layout == "head_major":
        last_len = int(last_k.shape[1])
        last_head_major = True
    else:
        raise ValueError(f"Unsupported last_layout: {last_layout}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    if not kv_head_major and kv_pages.shape != (
        num_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
    ):
        raise ValueError(
            "page-major kv_pages must have shape "
            f"({num_pages}, 2, {page_size}, {num_kv_heads}, {head_dim}), got {kv_pages.shape}"
        )
    if kv_head_major and kv_pages.shape != (
        num_kv_heads,
        num_pages,
        2,
        page_size,
        head_dim,
    ):
        raise ValueError(
            "head-major kv_pages must have shape "
            f"({num_kv_heads}, {num_pages}, 2, {page_size}, {head_dim}), got {kv_pages.shape}"
        )
    if not last_head_major and (
        last_k.shape != (last_len, num_kv_heads, head_dim)
        or last_v.shape != (last_len, num_kv_heads, head_dim)
    ):
        raise ValueError(
            "token-major last_k/last_v must have shape "
            f"({last_len}, {num_kv_heads}, {head_dim}), got {last_k.shape}/{last_v.shape}"
        )
    if last_head_major and (
        last_k.shape != (num_kv_heads, last_len, head_dim)
        or last_v.shape != (num_kv_heads, last_len, head_dim)
    ):
        raise ValueError(
            "head-major last_k/last_v must have shape "
            f"({num_kv_heads}, {last_len}, {head_dim}), got {last_k.shape}/{last_v.shape}"
        )
    if selected_count == 0 and last_len == 0:
        return mx.zeros((1, num_heads, 1, head_dim), dtype=q_2d.dtype)

    group_size = num_heads // num_kv_heads
    max_budget = max(1, selected_count)
    max_pages = _next_power_of_two(max(1, num_pages))
    threadgroup_size = 256
    kernel = _selected_indexed_page_decode_kernel(
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
            selected_pages.astype(mx.int32),
            kv_pages,
            last_k,
            last_v,
            num_heads,
            num_kv_heads,
            num_pages,
            selected_count,
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
            ("KV_HEAD_MAJOR", int(kv_head_major)),
            ("LAST_HEAD_MAJOR", int(last_head_major)),
        ],
        grid=(int(num_heads * threadgroup_size), 1, 1),
        threadgroup=(int(threadgroup_size), 1, 1),
        output_shapes=[(int(num_heads), int(head_dim))],
        output_dtypes=[q_2d.dtype],
        stream=mx.gpu,
    )[0]
    return out.reshape(1, num_heads, 1, head_dim)


@lru_cache(maxsize=64)
def _selected_frame_decode_kernel(
    head_dim: int,
    page_size: int,
    max_budget: int,
    group_size: int,
    max_frames: int,
    threadgroup_size: int,
):
    if head_dim not in (64, 128):
        raise ValueError(
            f"Metal selected-frame decode only supports head_dim 64/128, got {head_dim}"
        )
    if max_budget <= 0:
        raise ValueError("max_budget must be positive")
    return mx.fast.metal_kernel(
        name=(
            f"alayajet_selected_frame_decode_d{head_dim}_p{page_size}"
            f"_b{max_budget}_g{group_size}_f{max_frames}_t{threadgroup_size}"
        ),
        input_names=[
            "q",
            "selected_frames",
            "frames",
            "last_k",
            "last_v",
            "num_heads",
            "num_kv_heads",
            "num_frames",
            "selected_count",
            "last_len",
            "scale",
        ],
        output_names=["out"],
        source=_SELECTED_FRAME_DECODE_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=64)
def _selected_frame_with_last_kernel(
    head_dim: int,
    page_size: int,
    max_budget: int,
    group_size: int,
    max_frames: int,
    threadgroup_size: int,
):
    if head_dim not in (64, 128):
        raise ValueError(
            f"Metal selected-frame decode only supports head_dim 64/128, got {head_dim}"
        )
    return mx.fast.metal_kernel(
        name=(
            f"alayajet_selected_frame_with_last_d{head_dim}_p{page_size}"
            f"_b{max_budget}_g{group_size}_f{max_frames}_t{threadgroup_size}"
        ),
        input_names=[
            "q",
            "selected_frames",
            "frames",
            "num_heads",
            "num_kv_heads",
            "num_frames",
            "selected_count",
            "last_len",
            "scale",
        ],
        output_names=["out"],
        source=_SELECTED_FRAME_WITH_LAST_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=64)
def _selected_direct_frame_decode_kernel(
    head_dim: int,
    page_size: int,
    max_budget: int,
    group_size: int,
    max_direct_frames: int,
    threadgroup_size: int,
):
    if head_dim not in (64, 128):
        raise ValueError(
            f"Metal direct-frame decode only supports head_dim 64/128, got {head_dim}"
        )
    if max_budget <= 0:
        raise ValueError("max_budget must be positive")
    if max_direct_frames <= 0:
        raise ValueError("max_direct_frames must be positive")
    frame_names = [f"frame_{idx}" for idx in range(max_direct_frames)]
    return mx.fast.metal_kernel(
        name=(
            f"alayajet_selected_direct_frame_decode_d{head_dim}_p{page_size}"
            f"_b{max_budget}_g{group_size}_f{max_direct_frames}_t{threadgroup_size}"
        ),
        input_names=[
            "q",
            "selected_frames",
            *frame_names,
            "last_k",
            "last_v",
            "num_heads",
            "num_kv_heads",
            "selected_count",
            "last_len",
            "scale",
        ],
        output_names=["out"],
        source=_selected_direct_frame_source(max_direct_frames),
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=64)
def _selected_direct_frame_with_last_kernel(
    head_dim: int,
    page_size: int,
    max_budget: int,
    group_size: int,
    max_direct_frames: int,
    threadgroup_size: int,
):
    if head_dim not in (64, 128):
        raise ValueError(
            f"Metal direct-frame decode only supports head_dim 64/128, got {head_dim}"
        )
    frame_names = [f"frame_{idx}" for idx in range(max_direct_frames)]
    return mx.fast.metal_kernel(
        name=(
            f"alayajet_selected_direct_frame_with_last_d{head_dim}_p{page_size}"
            f"_b{max_budget}_g{group_size}_f{max_direct_frames}_t{threadgroup_size}"
        ),
        input_names=[
            "q",
            "selected_frames",
            *frame_names,
            "num_heads",
            "num_kv_heads",
            "selected_count",
            "last_len",
            "scale",
        ],
        output_names=["out"],
        source=_selected_direct_frame_with_last_source(max_direct_frames),
        ensure_row_contiguous=True,
    )


def selected_frame_decode_attention(
    q: mx.array,
    selected_frames: mx.array,
    frames: mx.array,
    last_k: mx.array,
    last_v: mx.array,
    *,
    scale: float,
    last_layout: str = "head_major",
) -> mx.array:
    """Metal decode attention that gathers selected pages by frame index.

    ``frames`` is a compact frame arena shaped
    ``(num_frames, 2, page_size, head_dim)``. ``selected_frames`` has shape
    ``(Hq, selected_count)`` and indexes that arena directly. This avoids the
    old selected-page path's per-KV-head page packing and padding.
    """
    if q.ndim == 4:
        q_2d = q[0, :, 0, :]
    elif q.ndim == 2:
        q_2d = q
    else:
        raise ValueError(f"Expected q rank 2 or 4, got shape {q.shape}")

    num_heads, head_dim = q_2d.shape
    if selected_frames.ndim != 2 or selected_frames.shape[0] != num_heads:
        raise ValueError(
            f"selected_frames must have shape ({num_heads}, selected_count), "
            f"got {selected_frames.shape}"
        )
    if frames.ndim != 4 or frames.shape[1] != 2:
        raise ValueError(
            "frames must have shape (num_frames, 2, page_size, head_dim), "
            f"got {frames.shape}"
        )
    num_frames = int(frames.shape[0])
    page_size = int(frames.shape[2]) if num_frames else 1
    if int(frames.shape[3]) != head_dim:
        raise ValueError(
            f"frames head_dim must be {head_dim}, got {frames.shape[3]}"
        )
    if last_layout == "head_major":
        num_kv_heads = int(last_k.shape[0])
        last_len = int(last_k.shape[1])
        last_head_major = True
    elif last_layout == "token_major":
        last_len = int(last_k.shape[0])
        num_kv_heads = int(last_k.shape[1])
        last_head_major = False
    else:
        raise ValueError(f"Unsupported last_layout: {last_layout}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    if last_head_major and (
        last_k.shape != (num_kv_heads, last_len, head_dim)
        or last_v.shape != (num_kv_heads, last_len, head_dim)
    ):
        raise ValueError(
            "head-major last_k/last_v must have shape "
            f"({num_kv_heads}, {last_len}, {head_dim}), got {last_k.shape}/{last_v.shape}"
        )
    if not last_head_major and (
        last_k.shape != (last_len, num_kv_heads, head_dim)
        or last_v.shape != (last_len, num_kv_heads, head_dim)
    ):
        raise ValueError(
            "token-major last_k/last_v must have shape "
            f"({last_len}, {num_kv_heads}, {head_dim}), got {last_k.shape}/{last_v.shape}"
        )

    selected_count = int(selected_frames.shape[1])
    if selected_count == 0 and last_len == 0:
        return mx.zeros((1, num_heads, 1, head_dim), dtype=q_2d.dtype)

    group_size = num_heads // num_kv_heads
    max_budget = max(1, selected_count)
    max_frames = _next_power_of_two(max(1, num_frames))
    threadgroup_size = 256
    kernel = _selected_frame_decode_kernel(
        head_dim,
        page_size,
        max_budget,
        group_size,
        max_frames,
        threadgroup_size,
    )
    out = kernel(
        inputs=[
            q_2d,
            selected_frames.astype(mx.int32),
            frames,
            last_k,
            last_v,
            num_heads,
            num_kv_heads,
            num_frames,
            selected_count,
            last_len,
            float(scale),
        ],
        template=[
            ("T", q_2d.dtype),
            ("HEAD_DIM", int(head_dim)),
            ("PAGE_SIZE", int(page_size)),
            ("MAX_BUDGET", int(max_budget)),
            ("GROUP_SIZE", int(group_size)),
            ("MAX_FRAMES", int(max_frames)),
            ("MAX_TOKENS", int(max_budget * page_size + page_size)),
            ("THREADGROUP_SIZE", int(threadgroup_size)),
            ("LANES_PER_DIM", int(threadgroup_size // head_dim)),
            ("LAST_HEAD_MAJOR", int(last_head_major)),
        ],
        grid=(int(num_heads * threadgroup_size), 1, 1),
        threadgroup=(int(threadgroup_size), 1, 1),
        output_shapes=[(int(num_heads), int(head_dim))],
        output_dtypes=[q_2d.dtype],
        stream=mx.gpu,
    )[0]
    return out.reshape(1, num_heads, 1, head_dim)


def selected_frame_decode_attention_with_last_frame(
    q: mx.array,
    selected_frames: mx.array,
    frames: mx.array,
    *,
    num_kv_heads: int,
    last_len: int,
    scale: float,
) -> mx.array:
    """Frame-arena decode where the active last page is included in selected_frames."""
    q_2d = q[0, :, 0, :] if q.ndim == 4 else q
    num_heads, head_dim = q_2d.shape
    if selected_frames.ndim != 2 or selected_frames.shape[0] != num_heads:
        raise ValueError(f"selected_frames shape mismatch: {selected_frames.shape}")
    if frames.ndim != 4 or frames.shape[1] != 2:
        raise ValueError(f"frames must have shape (N, 2, P, D), got {frames.shape}")
    page_size = int(frames.shape[2])
    if int(frames.shape[3]) != head_dim:
        raise ValueError("frame head_dim must match q head_dim")
    if num_heads % int(num_kv_heads) != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if last_len <= 0 or last_len > page_size:
        raise ValueError(f"last_len must be in [1, {page_size}], got {last_len}")
    selected_count = int(selected_frames.shape[1])
    group_size = num_heads // int(num_kv_heads)
    threadgroup_size = 256
    kernel = _selected_frame_with_last_kernel(
        head_dim,
        page_size,
        max(1, selected_count),
        group_size,
        int(frames.shape[0]),
        threadgroup_size,
    )
    out = kernel(
        inputs=[
            q_2d,
            selected_frames.astype(mx.int32),
            frames,
            num_heads,
            int(num_kv_heads),
            int(frames.shape[0]),
            selected_count,
            int(last_len),
            float(scale),
        ],
        template=[
            ("T", q_2d.dtype),
            ("HEAD_DIM", int(head_dim)),
            ("PAGE_SIZE", int(page_size)),
            ("MAX_BUDGET", int(selected_count)),
            ("GROUP_SIZE", int(group_size)),
            ("MAX_FRAMES", int(frames.shape[0])),
            ("MAX_TOKENS", int((selected_count - 1) * page_size + last_len)),
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


def selected_frame_decode_attention_with_stable_arena(
    q: mx.array,
    selected_frames: mx.array,
    arena: mx.array,
    *,
    num_kv_heads: int,
    last_len: int,
    scale: float,
) -> mx.array:
    """Decode attention over a stable per-layer resident arena.

    ``arena`` is shaped ``(max_blocks * num_kv_heads, 2, page_size, head_dim)``.
    ``selected_frames`` contains layer-local frame ids, so no selected-frame
    materialization is needed before launching the Metal kernel.
    """
    return selected_frame_decode_attention_with_last_frame(
        q,
        selected_frames,
        arena,
        num_kv_heads=num_kv_heads,
        last_len=last_len,
        scale=scale,
    )


def selected_direct_frame_decode_attention(
    q: mx.array,
    selected_frames: mx.array,
    frame_buffers: list[mx.array],
    last_k: mx.array,
    last_v: mx.array,
    *,
    scale: float,
    last_layout: str = "head_major",
) -> mx.array:
    """Metal decode attention that reads selected frame buffers directly.

    Unlike ``selected_frame_decode_attention``, this path does not stack frame
    buffers into a compact arena.  Each unique selected frame is bound as its
    own Metal input, and ``selected_frames`` indexes those inputs.  This is a
    bounded prototype for small page budgets; large selections should keep using
    the compact arena path or a true lower-level resident KV buffer.
    """
    if not frame_buffers:
        raise ValueError("frame_buffers must not be empty")
    if q.ndim == 4:
        q_2d = q[0, :, 0, :]
    elif q.ndim == 2:
        q_2d = q
    else:
        raise ValueError(f"Expected q rank 2 or 4, got shape {q.shape}")

    num_heads, head_dim = q_2d.shape
    if selected_frames.ndim != 2 or selected_frames.shape[0] != num_heads:
        raise ValueError(
            f"selected_frames must have shape ({num_heads}, selected_count), "
            f"got {selected_frames.shape}"
        )
    first_frame = frame_buffers[0]
    if first_frame.ndim != 3 or first_frame.shape[0] != 2:
        raise ValueError(
            "frame buffers must have shape (2, page_size, head_dim), "
            f"got {first_frame.shape}"
        )
    page_size = int(first_frame.shape[1])
    if int(first_frame.shape[2]) != head_dim:
        raise ValueError(
            f"frame buffer head_dim must be {head_dim}, got {first_frame.shape[2]}"
        )
    for frame in frame_buffers:
        if frame.shape != first_frame.shape:
            raise ValueError(
                f"all frame buffers must have shape {first_frame.shape}, got {frame.shape}"
            )
    if last_layout == "head_major":
        num_kv_heads = int(last_k.shape[0])
        last_len = int(last_k.shape[1])
        last_head_major = True
    elif last_layout == "token_major":
        last_len = int(last_k.shape[0])
        num_kv_heads = int(last_k.shape[1])
        last_head_major = False
    else:
        raise ValueError(f"Unsupported last_layout: {last_layout}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    selected_count = int(selected_frames.shape[1])
    if selected_count == 0 and last_len == 0:
        return mx.zeros((1, num_heads, 1, head_dim), dtype=q_2d.dtype)
    if selected_count == 0:
        raise ValueError("selected_direct_frame_decode_attention requires selected frames")

    group_size = num_heads // num_kv_heads
    max_budget = max(1, selected_count)
    max_direct_frames = len(frame_buffers)
    threadgroup_size = 256
    kernel = _selected_direct_frame_decode_kernel(
        head_dim,
        page_size,
        max_budget,
        group_size,
        max_direct_frames,
        threadgroup_size,
    )
    out = kernel(
        inputs=[
            q_2d,
            selected_frames.astype(mx.int32),
            *frame_buffers,
            last_k,
            last_v,
            num_heads,
            num_kv_heads,
            selected_count,
            last_len,
            float(scale),
        ],
        template=[
            ("T", q_2d.dtype),
            ("HEAD_DIM", int(head_dim)),
            ("PAGE_SIZE", int(page_size)),
            ("MAX_BUDGET", int(max_budget)),
            ("GROUP_SIZE", int(group_size)),
            ("MAX_TOKENS", int(max_budget * page_size + page_size)),
            ("THREADGROUP_SIZE", int(threadgroup_size)),
            ("LANES_PER_DIM", int(threadgroup_size // head_dim)),
            ("LAST_HEAD_MAJOR", int(last_head_major)),
        ],
        grid=(int(num_heads * threadgroup_size), 1, 1),
        threadgroup=(int(threadgroup_size), 1, 1),
        output_shapes=[(int(num_heads), int(head_dim))],
        output_dtypes=[q_2d.dtype],
        stream=mx.gpu,
    )[0]
    return out.reshape(1, num_heads, 1, head_dim)


def selected_direct_frame_decode_attention_with_last_frame(
    q: mx.array,
    selected_frames: mx.array,
    frame_buffers: list[mx.array],
    *,
    num_kv_heads: int,
    last_len: int,
    scale: float,
) -> mx.array:
    """Direct-frame decode where the active last page is just another frame id."""
    if not frame_buffers:
        raise ValueError("frame_buffers must not be empty")
    q_2d = q[0, :, 0, :] if q.ndim == 4 else q
    num_heads, head_dim = q_2d.shape
    if selected_frames.ndim != 2 or selected_frames.shape[0] != num_heads:
        raise ValueError(f"selected_frames shape mismatch: {selected_frames.shape}")
    first_frame = frame_buffers[0]
    page_size = int(first_frame.shape[1])
    if int(first_frame.shape[2]) != head_dim:
        raise ValueError("frame head_dim must match q head_dim")
    if num_heads % int(num_kv_heads) != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    selected_count = int(selected_frames.shape[1])
    if selected_count == 0:
        return mx.zeros((1, num_heads, 1, head_dim), dtype=q_2d.dtype)
    if last_len <= 0 or last_len > page_size:
        raise ValueError(f"last_len must be in [1, {page_size}], got {last_len}")
    group_size = num_heads // int(num_kv_heads)
    max_budget = max(1, selected_count)
    threadgroup_size = 256
    kernel = _selected_direct_frame_with_last_kernel(
        head_dim,
        page_size,
        max_budget,
        group_size,
        len(frame_buffers),
        threadgroup_size,
    )
    out = kernel(
        inputs=[
            q_2d,
            selected_frames.astype(mx.int32),
            *frame_buffers,
            num_heads,
            int(num_kv_heads),
            selected_count,
            int(last_len),
            float(scale),
        ],
        template=[
            ("T", q_2d.dtype),
            ("HEAD_DIM", int(head_dim)),
            ("PAGE_SIZE", int(page_size)),
            ("MAX_BUDGET", int(max_budget)),
            ("GROUP_SIZE", int(group_size)),
            ("MAX_TOKENS", int((selected_count - 1) * page_size + last_len)),
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
    kv_layout: str = "page_major",
    last_layout: str = "token_major",
) -> mx.array:
    """Fused Metal decode attention over paged KV candidates.

    Inputs:
      q: (1, Hq, 1, D) or (Hq, D)
      k_min/k_max: (Hkv, N, D), metadata aligned with ``kv_pages``.
      kv_pages: (N, 2, page_size, Hkv, D) when ``kv_layout='page_major'``,
        or (Hkv, N, 2, page_size, D) when ``kv_layout='head_major'``.
      last_k/last_v: (last_len, Hkv, D) when ``last_layout='token_major'``,
        or (Hkv, last_len, D) when ``last_layout='head_major'``.

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
    if kv_layout == "page_major":
        page_size = kv_pages.shape[2] if num_pages else 1
        kv_head_major = False
    elif kv_layout == "head_major":
        page_size = kv_pages.shape[3] if num_pages else 1
        kv_head_major = True
    else:
        raise ValueError(f"Unsupported kv_layout: {kv_layout}")
    if last_layout == "token_major":
        last_len = last_k.shape[0]
        last_head_major = False
    elif last_layout == "head_major":
        last_len = last_k.shape[1]
        last_head_major = True
    else:
        raise ValueError(f"Unsupported last_layout: {last_layout}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    if kv_head_major and kv_pages.shape[:2] != (num_kv_heads, num_pages):
        raise ValueError(
            "head-major kv_pages must have shape "
            f"({num_kv_heads}, {num_pages}, 2, page_size, head_dim), got {kv_pages.shape}"
        )
    if not kv_head_major and kv_pages.shape != (
        num_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
    ):
        raise ValueError(
            "page-major kv_pages must have shape "
            f"({num_pages}, 2, {page_size}, {num_kv_heads}, {head_dim}), got {kv_pages.shape}"
        )
    if last_head_major and (
        last_k.shape[:2] != (num_kv_heads, last_len)
        or last_v.shape[:2] != (
            num_kv_heads,
            last_len,
        )
    ):
        raise ValueError(
            "head-major last_k/last_v must have shape "
            f"({num_kv_heads}, {last_len}, head_dim), got {last_k.shape}/{last_v.shape}"
        )
    if not last_head_major and (
        last_k.shape != (last_len, num_kv_heads, head_dim)
        or last_v.shape != (last_len, num_kv_heads, head_dim)
    ):
        raise ValueError(
            "token-major last_k/last_v must have shape "
            f"({last_len}, {num_kv_heads}, {head_dim}), got {last_k.shape}/{last_v.shape}"
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
            ("KV_HEAD_MAJOR", int(kv_head_major)),
            ("LAST_HEAD_MAJOR", int(last_head_major)),
        ],
        grid=(int(num_heads * threadgroup_size), 1, 1),
        threadgroup=(int(threadgroup_size), 1, 1),
        output_shapes=[(int(num_heads), int(head_dim))],
        output_dtypes=[q_2d.dtype],
        stream=mx.gpu,
    )[0]
    return out.reshape(1, num_heads, 1, head_dim)
