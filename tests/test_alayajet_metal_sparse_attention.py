import unittest

import mlx.core as mx
import numpy as np

from alayajet.features.quest.metal_sparse_attention import (
    fused_sparse_decode_attention,
)


def _reference_sparse_decode(q, k_min, k_max, kv_pages, last_k, last_v, page_budget, scale):
    q = np.asarray(q, dtype=np.float32)
    k_min = np.asarray(k_min, dtype=np.float32)
    k_max = np.asarray(k_max, dtype=np.float32)
    kv_pages = np.asarray(kv_pages, dtype=np.float32)
    last_k = np.asarray(last_k, dtype=np.float32)
    last_v = np.asarray(last_v, dtype=np.float32)

    hq, head_dim = q.shape
    hkv, num_pages, _ = k_min.shape
    group_size = hq // hkv
    selected_count = min(page_budget, num_pages)
    out = np.zeros((hq, head_dim), dtype=np.float32)

    for h in range(hq):
        kv_h = h // group_size
        qh = q[h]
        estimate = np.sum(
            np.where(qh[None, :] >= 0.0, k_max[kv_h], k_min[kv_h]) * qh[None, :],
            axis=-1,
        )
        if selected_count:
            selected = np.argpartition(estimate, -selected_count)[-selected_count:]
        else:
            selected = np.empty((0,), dtype=np.int64)

        keys = []
        values = []
        for page in selected:
            keys.append(kv_pages[page, 0, :, kv_h, :])
            values.append(kv_pages[page, 1, :, kv_h, :])
        if last_k.shape[0]:
            keys.append(last_k[:, kv_h, :])
            values.append(last_v[:, kv_h, :])
        if not keys:
            continue
        k_full = np.concatenate(keys, axis=0)
        v_full = np.concatenate(values, axis=0)
        logits = (k_full @ qh) * scale
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        out[h] = probs @ v_full
    return out


@unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
class TestAlayaJetMetalSparseAttention(unittest.TestCase):
    def _run_fused_sparse_decode_case(self, *, head_dim, num_kv_heads=2):
        rng = np.random.default_rng(7)
        num_heads = 4
        num_pages = 6
        page_size = 8
        last_len = 5
        page_budget = 3
        scale = head_dim**-0.5

        q = rng.normal(size=(num_heads, head_dim)).astype(np.float16)
        kv_pages = rng.normal(
            size=(num_pages, 2, page_size, num_kv_heads, head_dim)
        ).astype(np.float16)
        last_k = rng.normal(size=(last_len, num_kv_heads, head_dim)).astype(np.float16)
        last_v = rng.normal(size=(last_len, num_kv_heads, head_dim)).astype(np.float16)
        k_values = kv_pages[:, 0]
        k_min = k_values.min(axis=1).transpose(1, 0, 2).astype(np.float32)
        k_max = k_values.max(axis=1).transpose(1, 0, 2).astype(np.float32)

        out = fused_sparse_decode_attention(
            mx.array(q).reshape(1, num_heads, 1, head_dim),
            mx.array(k_min),
            mx.array(k_max),
            mx.array(kv_pages),
            mx.array(last_k),
            mx.array(last_v),
            page_budget=page_budget,
            scale=scale,
        )
        mx.eval(out)
        ref = _reference_sparse_decode(
            q,
            k_min,
            k_max,
            kv_pages,
            last_k,
            last_v,
            page_budget,
            scale,
        )
        np.testing.assert_allclose(np.asarray(out[0, :, 0, :]), ref, rtol=2e-2, atol=2e-2)

    def test_fused_sparse_decode_matches_reference_gqa_d64(self):
        self._run_fused_sparse_decode_case(head_dim=64)

    def test_fused_sparse_decode_matches_reference_gqa_d128(self):
        self._run_fused_sparse_decode_case(head_dim=128)

    def test_fused_sparse_decode_matches_reference_mqa_d64(self):
        self._run_fused_sparse_decode_case(head_dim=64, num_kv_heads=1)


if __name__ == "__main__":
    unittest.main()
